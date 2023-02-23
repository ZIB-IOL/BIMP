# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         strategies/scratchStrategies.py
# Description:  Sparsification strategies for regular training
# ===========================================================================
from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune


# Dense Base Class
class Dense:
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits. Should be
    used when simply training a model."""
    required_params = []    # Specifies the hyperparameters required for filtering the pretrained runs later

    def __init__(self, **kwargs):
        self.masks = dict()
        self.lr_dict = OrderedDict()  # it:lr
        self.is_in_finetuning_phase = False

        self.model = kwargs['model']
        self.run_config = kwargs['config']
        self.callbacks = kwargs['callbacks']
        self.goal_sparsity = self.run_config['goal_sparsity']

        # Variables to be set later
        self.optimizer = None
        self.n_total_iterations = None
        self.parameters_to_prune = None
        self.n_prunable_parameters = None

    def after_initialization(self):
        """Called after initialization of the strategy"""
        self.parameters_to_prune = [(module, 'weight') for name, module in self.model.named_modules() if
                                    hasattr(module, 'weight')
                                    and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                                                     torch.nn.BatchNorm2d)]
        self.n_prunable_parameters = sum(
            getattr(module, param_type).numel() for module, param_type in self.parameters_to_prune)

    def set_optimizer(self, opt, **kwargs):
        """Sets the optimizer to be used for training."""
        self.optimizer = opt
        if 'n_total_iterations' in kwargs:
            self.n_total_iterations = kwargs['n_total_iterations']

    @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Function to be called before Forward step."""
        pass

    @torch.no_grad()
    def end_forward_mode(self, **kwargs):
        """Function to be called after Forward step."""
        pass

    @torch.no_grad()
    def before_backward(self, **kwargs):
        """Function to be called after Forward step. Should return loss if it is not modified."""
        return kwargs['loss']

    @torch.no_grad()
    def during_training(self, **kwargs):
        """Function to be called after loss.backward() and before optimizer.step(), e.g. to mask gradients."""
        pass

    @torch.no_grad()
    def after_training_iteration(self, **kwargs):
        """Called after each training iteration"""
        if not self.is_in_finetuning_phase:
            self.lr_dict[kwargs['it']] = kwargs['lr']

    def at_train_begin(self):
        """Called before training begins"""
        pass

    def at_epoch_start(self, **kwargs):
        """Called before the epoch starts"""
        pass

    def at_epoch_end(self, **kwargs):
        """Called at epoch end"""
        pass

    def at_train_end(self, **kwargs):
        """Called at the end of training"""
        pass

    def final(self):
        """Called at the very end."""
        self.make_pruning_permanent()

    @torch.no_grad()
    def pruning_step(self, pruning_sparsity: float, only_save_mask: bool = False, compute_from_scratch: bool = False):
        """Prunes the model to the given sparsity.
         If only_save_mask is True, the mask is saved but not applied.
         If compute_from_scratch is True, the mask is computed from scratch, otherwise existing masks are extended."""
        if compute_from_scratch:
            # We have to revert to weight_orig and then compute the mask
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Enforce the equivalence of weight_orig and weight
                    orig = getattr(module, param_type + "_orig").detach().clone()
                    prune.remove(module, param_type)
                    p = getattr(module, param_type)
                    p.copy_(orig)
                    del orig
        elif only_save_mask and len(self.masks) > 0:
            for module, param_type in self.parameters_to_prune:
                if (module, param_type) in self.masks:
                    prune.custom_from_mask(module, param_type, self.masks[(module, param_type)])

        # Default: prune globally
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=self.get_pruning_method(),
            amount=pruning_sparsity,
        )

        self.masks = dict()  # Stays empty if we use regular pruning
        if only_save_mask:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Save the mask
                    mask = getattr(module, param_type + '_mask')
                    self.masks[(module, param_type)] = mask.detach().clone()
                    setattr(module, param_type + '_mask', torch.ones_like(mask))
                    # Remove (i.e. make permanent) the reparameterization
                    prune.remove(module=module, name=param_type)

    def enforce_prunedness(self):
        """
        Makes the pruning permanent, i.e. set the pruned weights to zero, than reinitialize from the same mask
        This ensures that we can actually work (i.e. LMO, rescale computation) with the parameters
        """
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Save the mask
                mask = getattr(module, param_type + '_mask')
                # Remove (i.e. make permanent) the reparameterization
                prune.remove(module=module, name=param_type)
                # Reinitialize the pruning
                prune.custom_from_mask(module=module, name=param_type, mask=mask)

    def prune_momentum(self):
        """Prunes the momentum buffer of the optimizer"""
        opt_state = self.optimizer.state
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the prunedness of momentum buffer
                param_state = opt_state[getattr(module, param_type + "_orig")]
                if 'momentum_buffer' in param_state:
                    mask = getattr(module, param_type + "_mask")
                    param_state['momentum_buffer'] *= mask.to(dtype=param_state['momentum_buffer'].dtype)

    def get_pruning_method(self):
        raise NotImplementedError("Dense has no pruning method, this must be implemented in each child class.")

    @torch.no_grad()
    def make_pruning_permanent(self):
        """Makes the pruning permanent and removes the pruning hooks"""
        if len(self.masks) == 0:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    prune.remove(module, param_type)
        else:
            for module, param_type in self.masks:
                # Get the mask
                mask = self.masks[(module, param_type)]

                # Apply the mask
                orig = getattr(module, param_type)
                orig *= mask
            self.masks = dict()

    def set_to_finetuning_phase(self):
        self.is_in_finetuning_phase = True


# Pruning stable strategies
class LC(Dense):
    """Learning compression as in Carreira-Perpinan et al. (2018)"""
    required_params = ['goal_sparsity']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_remaining_params = None  # Number of parameters that remain after pruning

    def after_initialization(self):
        super().after_initialization()
        self.n_remaining_params = int((1 - self.goal_sparsity) * self.n_prunable_parameters)

    @torch.no_grad()
    def during_training(self, **kwargs) -> None:
        """Modify gradient such that only the n-k smallest weights are decayed. This is done by adding to the gradient
        such that regular weight decay cancels out on the k largest weights."""
        param_list = [p for group in self.optimizer.param_groups
                      for p in group['params'] if p.grad is not None]
        # Get the vector
        param_vector = torch.cat([p.view(-1) for p in param_list])
        param_vector_shape = param_vector.shape
        device = param_list[0].device
        top_indices = torch.topk(torch.abs(param_vector), k=self.n_remaining_params).indices
        update_vector = torch.zeros(param_vector_shape, device=device)
        update_vector[top_indices] = param_vector[top_indices]
        del param_vector
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        for p in param_list:
            numberOfElements = p.numel()
            partial_update_vector = update_vector[:numberOfElements].view(p.shape)
            update_vector = update_vector[numberOfElements:]
            p.grad.add_(partial_update_vector, alpha=-weight_decay)


class GSM(Dense):
    """Global Sparse Momentum as by Ding et al. (2019)."""
    required_params = ['goal_sparsity', 'momentum']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_remaining_params = None  # Number of parameters to remain after pruning

    def after_initialization(self):
        super().after_initialization()
        self.n_remaining_params = int((1 - self.goal_sparsity) * self.n_prunable_parameters)

    @torch.no_grad()
    def during_training(self, **kwargs) -> None:
        """Apply top_k mask to the gradients"""
        param_list = [p for group in self.optimizer.param_groups
                      for p in group['params'] if p.grad is not None]
        # Get the vector
        grad_vector = torch.cat([torch.abs(p * p.grad).view(-1) for p in param_list])
        grad_vector_shape = grad_vector.shape
        device = param_list[0].device
        top_indices = torch.topk(grad_vector, k=self.n_remaining_params).indices
        del grad_vector
        mask_vector = torch.zeros(grad_vector_shape, device=device)
        mask_vector[top_indices] = 1

        for p in param_list:
            numberOfElements = p.numel()
            partial_mask = mask_vector[:numberOfElements].view(p.shape)
            mask_vector = mask_vector[numberOfElements:]
            p.grad.mul_(partial_mask)  # Mask gradient


class GMP(Dense):
    """Gradual Magnitude Pruning as proposed by Zhu & Gupta (2017)."""
    required_params = ['pruning_interval', 'allow_recovering', 'goal_sparsity']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_interval = self.run_config['pruning_interval']
        self.allow_recovering = self.run_config['allow_recovering']
        self.n_train_epochs = self.run_config['n_epochs']
        assert 2 * self.pruning_interval <= self.n_train_epochs, "Pruning interval too large."

        self.current_sparsity = 0.0
        self.n_pruning_steps = self.n_train_epochs // self.pruning_interval - 1
        self.pruning_epochs = OrderedDict(
            {int(round(t * self.pruning_interval)): self.sparsity_schedule(int(round(t * self.pruning_interval))) for t
             in
             range(1, self.n_pruning_steps + 1, 1)})
        self.currently_required_sparsity = self.current_sparsity

    def sparsity_schedule(self, t):
        """Sparsity schedule as in Zhu & Gupta (2017)"""
        return self.goal_sparsity + (0 - self.goal_sparsity) * (
                1 - t / int(round(self.pruning_interval * self.n_pruning_steps))) ** 3

    def at_epoch_end(self, **kwargs):
        """Prune at the specified epochs"""
        epoch = kwargs['epoch']
        super().at_epoch_end(**kwargs)
        did_prune = self.pruning_scheduler(epoch=epoch)
        if did_prune:
            # We need to prune momentum, otherwise the non-active weights are changed
            self.prune_momentum()

    def pruning_scheduler(self, epoch):
        """Check if we need to prune at the current epoch. If so, prune and return True. Otherwise, return False."""
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            current_density = 1 - self.current_sparsity
            sparsity_step = 1. - (1 - self.currently_required_sparsity) / current_density
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(pruning_sparsity=sparsity_step, compute_from_scratch=self.allow_recovering)
            if self.allow_recovering and epoch != list(self.pruning_epochs.keys())[-1]:
                self.current_sparsity = 0.0
            return True
        return False

    def at_train_end(self, **kwargs):
        """Make the pruning permanent at the end of training to get rid of the masks."""
        self.make_pruning_permanent()

    def get_pruning_method(self):
        return prune.L1Unstructured


class DPF(GMP):
    """Dynamic Pruning with Feedback as proposed by Lin et al. (2020)"""
    required_params = ['pruning_interval', 'goal_sparsity']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_parameters = dict()  # Saves the error between before_forward and after_forward

    @torch.no_grad()
    def enforce_prunedness(self):
        """Overwrite it such that weight and weight_orig are not enforced to be the same"""
        pass

    @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Modify weights to use pruned version in forward step, but accumulate gradients of pruned weights nonetheless.
        This has to be reversed by during_training before updating the parameters."""
        if len(self.masks) == 0:
            return

        for module, param_type in self.masks:
            # Get the mask
            mask = self.masks[(module, param_type)]

            # Change the weights to incorporate error feedback
            orig = getattr(module, param_type)
            self.original_parameters[(module, param_type)] = orig.detach().clone()
            orig *= mask

    @torch.no_grad()
    def end_forward_mode(self, **kwargs):
        """Same call as during_training"""
        self.during_training(**kwargs)

    @torch.no_grad()
    def during_training(self, **kwargs):
        """Reset weights to original status"""
        if len(self.original_parameters) == 0:
            return
        for (module, param_type) in self.original_parameters:
            p = getattr(module, param_type)
            p.copy_(self.original_parameters[(module, param_type)])
        self.original_parameters = dict()

    def pruning_scheduler(self, epoch):
        """Prune at the specified epochs."""
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(pruning_sparsity=self.currently_required_sparsity, only_save_mask=True,
                              compute_from_scratch=True)
            return True
        return False

    @torch.no_grad()
    def after_training_iteration(self, **kwargs):
        """Called after each training iteration. We prune every 16 iterations, as in the original paper."""
        super().after_training_iteration(**kwargs)
        if kwargs['it'] % 16 == 0:
            self.pruning_step(pruning_sparsity=self.currently_required_sparsity, only_save_mask=True,
                              compute_from_scratch=True)


class DNW(Dense):
    """Discovering Neural Wirings as in Wortsman et al. (2019)."""
    required_params = ['goal_sparsity']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        class ChooseEdges(torch.autograd.Function):
            # Adapted from https://github.com/RAIVNLab/STR
            @staticmethod
            def forward(ctx, weight, thresh):
                output = weight * torch.where(torch.abs(weight) > thresh, 1., 0.)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # Straight through
                return grad_output, None

        self.prune_fn = ChooseEdges.apply

    def after_initialization(self):
        """Called after initialization of the strategy"""
        super().after_initialization()
        self.register_masks()

    # Important: no @torch.no_grad()
    def apply_reparameterization(self, p, thresh):
        res = self.prune_fn(p, thresh)
        return res

    @torch.no_grad()
    def register_masks(self) -> None:
        """Copy parameter for every param in self.parameters_to_prune"""
        for module, param_type in self.parameters_to_prune:
            orig = getattr(module, param_type)
            device = orig.device

            # Make sure this is not a parameter of the model anymore
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(param_type + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[param_type]
            new_tensor = orig.detach().clone()
            new_tensor = new_tensor.to(device=device)
            setattr(module, param_type, new_tensor)

    @torch.no_grad()
    def deregister_masks(self, opt) -> None:
        with torch.no_grad():
            param_vector = torch.cat(
                [getattr(module, param_type + "_orig").view(-1) for module, param_type in self.parameters_to_prune
                 if hasattr(module, param_type + '_orig')])
            n_prune_global = int(self.goal_sparsity * self.n_prunable_parameters)  # Number of parameters to prune
            # Get the (n_prune_global)th smallest entry, prune everything below it
            threshold = torch.kthvalue(torch.abs(param_vector), k=n_prune_global).values

        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_orig"):
                orig = getattr(module, param_type + "_orig")

                # Compute the pruning mask
                binary_mask = torch.where(torch.abs(orig) > threshold, 1, 0)
                # Set original parameters before pruning
                setattr(module, param_type, torch.nn.Parameter(orig))

                # Make .weight/.bias the Parameter, delete the rest
                module.register_parameter(param_type, getattr(module, param_type))
                del module._parameters[param_type + "_orig"]

                # Prune from learned binary mask
                prune.custom_from_mask(module, param_type, binary_mask)

        # Reset optimizer parameter
        opt.param_groups[0]['params'] = list(self.model.parameters())

    # Important: no @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Apply reparameterization in the forward and backward pass,
         no reverting needed except at the end of training"""
        if 'enable_grad' not in kwargs:
            enable_grad = False
        else:
            enable_grad = kwargs['enable_grad']
        torch.set_grad_enabled(enable_grad)
        with torch.no_grad():
            param_list = [getattr(module, param_type + "_orig").view(-1) for module, param_type in
                          self.parameters_to_prune
                          if hasattr(module, param_type + '_orig') and not hasattr(module, param_type + '_mask')]
            if len(param_list) > 0:
                param_vector = torch.cat(param_list).to(device=param_list[0].device)
                n_prune_global = int(
                    self.goal_sparsity * self.n_prunable_parameters)  # Number of parameters to prune
                # Get the (n_prune_global)th smallest entry, prune everything below it
                threshold = torch.kthvalue(torch.abs(param_vector), k=n_prune_global).values
                del param_vector
            else:
                threshold = 0.
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_orig") and not hasattr(module, param_type + "_mask"):
                orig = getattr(module, param_type + "_orig")
                # Clear potential gradients
                getattr(module, param_type).detach_()

                # Forward operation of reparameterization
                res = self.apply_reparameterization(p=orig, thresh=threshold)
                setattr(module, param_type, res)

    def at_train_end(self):
        self.deregister_masks(opt=self.optimizer)  # This is equivalent to pruning


class STR(Dense):
    """Soft Threshold Weight Reparameterization as proposed by Kusupati et al. (2020)."""
    required_params = ['s_initial']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.s_initial = self.run_config['s_initial']
        self.g = torch.sigmoid

    def after_initialization(self):
        """Called after initialization of the strategy"""
        super().after_initialization()
        self.register_masks()

    # Important: no @torch.no_grad()
    def apply_reparameterization(self, p, thresh):
        res = torch.sign(p) * torch.relu(torch.abs(p) - self.g(thresh))
        return res

    @torch.no_grad()
    def register_masks(self) -> None:
        """Add a learnable mask parameter for every param in self.parameters_to_prune"""
        for module, param_type in self.parameters_to_prune:
            orig = getattr(module, param_type)
            # Create thresh and _orig tensors
            p_thresh = torch.nn.Parameter(torch.tensor(float(self.s_initial), device=orig.device),
                                          requires_grad=True)
            module.register_parameter(param_type + "_thresh", p_thresh)

            # Make sure this is not a parameter of the model anymore
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(param_type + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[param_type]
            setattr(module, param_type, orig.detach().clone())

    @torch.no_grad()
    def deregister_masks(self, opt) -> None:
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh"):
                thresh = getattr(module, param_type + "_thresh")
                orig = getattr(module, param_type + "_orig")

                # Get sparse weights. Note: the final weights include the -threshold term for non-pruned weights
                # In other words: it is not safe to just prune using the thresholds
                final = self.apply_reparameterization(p=orig, thresh=thresh)  # Ensures that correct values are used
                binary_mask = torch.where(torch.abs(orig) > self.g(thresh), 1, 0)
                # Set original parameters before pruning
                setattr(module, param_type, torch.nn.Parameter(final))

                # Make .weight/.bias the Parameter, delete the rest
                module.register_parameter(param_type, getattr(module, param_type))

                del module._parameters[param_type + "_thresh"]
                del module._parameters[param_type + "_orig"]

                # Prune from learned binary mask
                prune.custom_from_mask(module, param_type, binary_mask)

        # Reset optimizer parameter
        opt.param_groups[0]['params'] = list(self.model.parameters())

    # Important: no @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Apply reparameterization in the forward and backward pass,
         no reverting needed except at the end of training"""
        if 'enable_grad' not in kwargs:
            enable_grad = False
        else:
            enable_grad = kwargs['enable_grad']
        torch.set_grad_enabled(enable_grad)
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh"):
                thresh = getattr(module, param_type + "_thresh")
                orig = getattr(module, param_type + "_orig")
                # Clear potential gradients
                getattr(module, param_type).detach_()

                # Forward operation of reparameterization
                res = self.apply_reparameterization(p=orig, thresh=thresh)
                setattr(module, param_type, res)

    @torch.no_grad()
    def end_forward_mode(self, **kwargs):
        """Do nothing at all"""
        pass

    def at_train_end(self, **kwargs):
        self.deregister_masks(opt=self.optimizer)  # This is equivalent to pruning


class CS(Dense):
    """Continuous Sparsification as proposed by Savarese et al. (2019)."""
    required_params = ['s_initial', 'beta_final', 'penalty']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.s_initial = self.run_config['s_initial']
        self.beta_final = self.run_config['beta_final']
        self.penalty = self.run_config['penalty']

        self.original_parameters = dict()
        self.beta_current = 1
        self.sigma = torch.sigmoid
        self.scaling_factor = float(
            1. / self.sigma(torch.tensor(float(self.s_initial))))  # Rescale forward as is done in their implementation

    def after_initialization(self):
        """Called after initialization of the strategy"""
        super().after_initialization()
        self.register_masks()

    # Important: no @torch.no_grad()
    def apply_reparameterization(self, p, mask):
        res = self.scaling_factor * p * self.sigma(
            self.beta_current * mask)  # self.sigma(beta*mask) is the actual mask used
        return res

    @torch.no_grad()
    def register_masks(self) -> None:
        """Add a learnable mask parameter for every param in self.parameters_to_prune"""
        for module, param_type in self.parameters_to_prune:
            orig = getattr(module, param_type)
            # Create mask and _orig tensors
            p_mask = torch.nn.Parameter(self.s_initial * torch.ones_like(orig), requires_grad=True)

            # Make sure this is not a parameter of the model anymore
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(param_type + "_mask", p_mask)
            module.register_parameter(param_type + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[param_type]
            setattr(module, param_type, orig.detach().clone())

    @torch.no_grad()
    def deregister_masks(self, opt) -> None:
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_mask") and hasattr(module, param_type + "_orig"):
                mask = getattr(module, param_type + "_mask")
                orig = getattr(module, param_type + "_orig")

                final = self.apply_reparameterization(p=orig, mask=mask)  # Ensures that correct values are used
                # Apply heaviside to mask to get binary mask
                binary_mask = torch.where(mask > 0, 1, 0)
                # Set original parameters before pruning
                setattr(module, param_type, torch.nn.Parameter(final))

                # Make .weight/.bias the Parameter, delete the rest
                module.register_parameter(param_type, getattr(module, param_type))
                del module._parameters[param_type + "_mask"]
                del module._parameters[param_type + "_orig"]

                # Prune from learned binary mask
                prune.custom_from_mask(module, param_type, binary_mask)

        # Reset optimizer parameter
        opt.param_groups[0]['params'] = list(self.model.parameters())

    # Important: no @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Apply reparameterization in the forward and backward pass, no reverting needed except at the end of training"""
        if 'enable_grad' not in kwargs:
            enable_grad = False
        else:
            enable_grad = kwargs['enable_grad']
        torch.set_grad_enabled(enable_grad)
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_mask") and hasattr(module, param_type + "_orig"):
                mask = getattr(module, param_type + "_mask")
                orig = getattr(module, param_type + "_orig")
                # Clear potential gradients
                getattr(module, param_type).detach_()

                # Forward operation of reparameterization
                res = self.apply_reparameterization(p=orig, mask=mask)
                setattr(module, param_type, res)

    @torch.no_grad()
    def end_forward_mode(self, **kwargs):
        """Do nothing at all"""
        pass

    # Important: no torch.no_grad
    def before_backward(self, **kwargs):
        """Add penalty"""
        loss = kwargs['loss']
        wd = kwargs['weight_decay']
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_mask") and hasattr(module, param_type + "_orig"):
                orig = getattr(module, param_type + "_orig")
                mask = getattr(module, param_type + "_mask")
                loss = loss + self.penalty * torch.sum(
                    self.sigma(self.beta_current * mask))  # abs not needed since result is positive
                loss = loss + 0.5 * wd * torch.sum(orig ** 2)
        return loss

    @torch.no_grad()
    def after_training_iteration(self, **kwargs):
        """Called after each training iteration"""
        super().after_training_iteration(**kwargs)
        it = kwargs['it']
        exponent = float(it) / self.n_total_iterations
        self.beta_current = self.beta_final ** exponent

    def at_train_end(self, **kwargs):
        self.deregister_masks(opt=self.optimizer)  # This is equivalent to pruning


class DST(Dense):
    """Dynamic Sparse Training as in Liu et al. (2020)"""
    required_params = ['penalty']

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.penalty = self.run_config['penalty']

        class BinaryStep(torch.autograd.Function):
            """BinaryStep function from https://github.com/junjieliu2910/DynamicSparseTraining"""

            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return (input > 0.).float()

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                zero_index = torch.abs(input) > 1
                middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
                additional = 2 - 4 * torch.abs(input)
                additional[zero_index] = 0.
                additional[middle_index] = 0.4
                return grad_input * additional

        self.heaviside = BinaryStep.apply

    def after_initialization(self):
        """Called after initialization of the strategy"""
        super().after_initialization()
        self.register_masks()

    # Important: no @torch.no_grad()
    def apply_reparameterization(self, p, thresh):
        p_shape = p.shape
        abs_weight = torch.abs(p).view(p_shape[0], -1)
        threshold_view = thresh.view(p_shape[0], -1)
        abs_weight = abs_weight - threshold_view
        mask = self.heaviside(abs_weight)
        ratio = torch.sum(mask) / float(mask.numel())
        if ratio <= 0.01:
            with torch.no_grad():
                thresh.data.fill_(0)
            abs_weight = torch.abs(p).view(p_shape[0], -1)
            threshold_view = thresh.view(p_shape[0], -1)
            abs_weight = abs_weight - threshold_view
            mask = self.heaviside(abs_weight)

        mask = mask.view(p_shape)
        masked_weight = p * mask
        return masked_weight

    @torch.no_grad()
    def register_masks(self) -> None:
        """Add a learnable mask parameter for every param in self.parameters_to_prune"""
        for module, param_type in self.parameters_to_prune:
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                orig = getattr(module, param_type)
                if isinstance(module, torch.nn.Linear):
                    dim = getattr(module, 'out_features')
                elif isinstance(module, torch.nn.Conv2d):
                    dim = getattr(module, 'out_channels')
                # Create thresh and _orig tensors
                # Zero initialization
                p_thresh = torch.nn.Parameter(torch.empty(dim, device=orig.device).fill_(0.), requires_grad=True)
                module.register_parameter(param_type + "_thresh", p_thresh)

                # Make sure this is not a parameter of the model anymore
                # copy `module[name]` to `module[name + '_orig']`
                module.register_parameter(param_type + "_orig", orig)
                # temporarily delete `module[name]`
                del module._parameters[param_type]
                setattr(module, param_type, orig.detach().clone())

    @torch.no_grad()
    def deregister_masks(self, opt) -> None:
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh"):
                thresh = getattr(module, param_type + "_thresh")
                orig = getattr(module, param_type + "_orig")

                orig_shape = orig.shape
                abs_weight = torch.abs(orig).view(orig_shape[0], -1)
                threshold_view = thresh.view(orig_shape[0], -1)
                abs_weight = abs_weight - threshold_view
                binary_mask = self.heaviside(abs_weight).view(orig_shape)
                # Set original parameters before pruning
                setattr(module, param_type, torch.nn.Parameter(orig))

                # Make .weight/.bias the Parameter, delete the rest
                module.register_parameter(param_type, getattr(module, param_type))
                del module._parameters[param_type + "_thresh"]
                del module._parameters[param_type + "_orig"]

                # Prune from learned binary mask
                prune.custom_from_mask(module, param_type, binary_mask)

        # Reset optimizer parameter
        opt.param_groups[0]['params'] = list(self.model.parameters())

    # Important: no @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Apply reparameterization in the forward and backward pass, no reverting needed except at the end of training"""
        if 'enable_grad' not in kwargs:
            enable_grad = False
        else:
            enable_grad = kwargs['enable_grad']
        torch.set_grad_enabled(enable_grad)
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh"):
                thresh = getattr(module, param_type + "_thresh")
                orig = getattr(module, param_type + "_orig")
                # Clear potential gradients
                getattr(module, param_type).detach_()

                # Forward operation of reparameterization
                res = self.apply_reparameterization(p=orig, thresh=thresh)
                setattr(module, param_type, res)

    # Important: no torch.no_grad
    def before_backward(self, **kwargs):
        """Add penalty"""
        loss = kwargs['loss']
        wd = kwargs['weight_decay']
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh") and hasattr(module, param_type + "_orig"):
                orig = getattr(module, param_type + "_orig")
                thresh = getattr(module, param_type + "_thresh")
                loss = loss + self.penalty * torch.sum(torch.exp(-thresh))
                loss = loss + 0.5 * wd * torch.sum(orig ** 2)
        return loss

    def at_train_end(self, **kwargs):
        self.deregister_masks(opt=self.optimizer)  # This is equivalent to pruning
