# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         strategies/strategies.py
# Description:  Sparsification strategies
# ===========================================================================

from collections import OrderedDict
from math import ceil, log

import torch
import torch.nn.utils.prune as prune

import metrics.metrics
from utilities import LAMPUnstructured


#### Basic & IMP-based Strategies
class Dense:
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits."""

    def __init__(self):
        self.masks = dict()
        self.lr_dict = OrderedDict()  # epoch:lr
        self.train_loss_dict = OrderedDict()  # epoch:train_loss
        self.train_acc_dict = OrderedDict()
        self.is_in_finetuning_phase = False

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if
                                    hasattr(module, 'weight')
                                    and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                                                     torch.nn.BatchNorm2d)]
        self.n_prunable_parameters = sum(
            getattr(module, param_type).numel() for module, param_type in self.parameters_to_prune)

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
        """Function to be called after Forward step. Should return loss also if it is not modified."""
        return kwargs['loss']

    @torch.no_grad()
    def during_training(self, **kwargs):
        """Function to be called after loss.backward() and before optimizer.step, e.g. to mask gradients."""
        pass

    @torch.no_grad()
    def after_training_iteration(self, it):
        """Called after each training iteration"""
        pass

    def at_train_begin(self, model, LRScheduler):
        pass

    def at_epoch_end(self, **kwargs):
        self.lr_dict[kwargs['epoch']] = kwargs['epoch_lr']
        self.train_loss_dict[kwargs['epoch']] = kwargs['train_loss']
        self.train_acc_dict[kwargs['epoch']] = kwargs['train_acc']

    def at_train_end(self, **kwargs):
        pass

    def final(self, model, final_log_callback):
        self.remove_pruning_hooks(model=model)

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
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

        # Do not prune biases and batch norm
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
                    # Delete the temporary mask to free memory
                    del mask

    def prune_momentum(self, optimizer):
        opt_state = optimizer.state
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the prunedness of momentum buffer
                param_state = opt_state[getattr(module, param_type + "_orig")]
                if 'momentum_buffer' in param_state:
                    mask = getattr(module, param_type + "_mask")
                    param_state['momentum_buffer'] *= mask.to(dtype=param_state['momentum_buffer'].dtype)

    def reset_momentum(self, optimizer):
        opt_state = optimizer.state
        for group in optimizer.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = opt_state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def enforce_prunedness(self):
        """Secures that weight_orig has the same entries as weight. Important: At the moment of execution
        .weight might be old, it is updated in the forward pass automatically. Hence we update it by explicit mask application."""
        return

    @torch.no_grad()
    def get_per_layer_thresholds(self):
        """If properly defined, returns a list of per layer thresholds"""
        return []

    def get_pruning_method(self):
        return prune.Identity

    @torch.no_grad()
    def remove_pruning_hooks(self, model):
        # Note: this does not remove the pruning itself, but rather makes it permanent
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

    def initial_prune(self):
        pass

    def set_to_finetuning_phase(self):
        self.is_in_finetuning_phase = True


class IMP(Dense):
    def __init__(self, desired_sparsity: float, n_phases: int = 1, n_epochs_per_phase: int = 1) -> None:
        super().__init__()
        self.desired_sparsity = desired_sparsity
        self.n_phases = n_phases  # If None, compute this manually using Renda's approach of pruning 20% per phase
        self.n_epochs_per_phase = n_epochs_per_phase
        # Sparsity factor on remaining weights after each round, yields desired_sparsity after all rounds
        if self.n_phases is not None:
            self.pruning_sparsity = 1 - (1 - self.desired_sparsity) ** (1. / self.n_phases)
        else:
            self.pruning_sparsity = 0.2
            self.n_phases = ceil(log(1 - self.desired_sparsity, 1 - self.pruning_sparsity))

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        restore_callback()  # Restore to checkpoint model
        prune_per_phase = self.pruning_sparsity
        for phase in range(1, self.n_phases + 1, 1):
            self.pruning_step(model, pruning_sparsity=prune_per_phase)
            self.current_sparsity = 1 - (1 - prune_per_phase) ** phase
            after_pruning_callback(desired_sparsity=self.current_sparsity)
            self.finetuning_step(desired_sparsity=self.current_sparsity, finetuning_callback=finetuning_callback,
                                 phase=phase)
            save_model_callback(
                model_type=f"{self.current_sparsity}-sparse_final")  # removing of pruning hooks happens in restore_callback

    def finetuning_step(self, desired_sparsity, finetuning_callback, phase):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase,
                            phase=phase)

    def get_pruning_method(self):
        return prune.L1Unstructured

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()


class IMP_LAMP(IMP):
    """LAMP criterion to select local sparsities (Lee2020)."""

    def __init__(self, **kwargs):
        super(IMP_LAMP, self).__init__(**kwargs)

    def get_pruning_method(self):
        intermediate_pruning_class = lambda amount: LAMPUnstructured(parameters_to_prune=self.parameters_to_prune,
                                                                     amount=amount)
        return intermediate_pruning_class


class IMP_Uniform(IMP):
    """Uniform sparsity distribution among layers."""

    def __init__(self, **kwargs):
        super(IMP_Uniform, self).__init__(**kwargs)

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
        for module, param_type in self.parameters_to_prune:
            prune.l1_unstructured(module, name=param_type, amount=pruning_sparsity)


class IMP_UniformPlus(IMP):
    """Uniform sparsity distribution among layers, but keep the first Convlayer dense and the last FC-layer with at least 20% weights remaining (Proposed by Gale et al. 2019)."""

    def __init__(self, **kwargs):
        super(IMP_UniformPlus, self).__init__(**kwargs)
        assert kwargs['n_phases'] == 1, 'Does not work with multiple phases at this point.'

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
        # Make permanent
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, param_type)

        masks = self.compute_custom_masks(pruning_sparsity=pruning_sparsity)
        for module, param_type in masks:
            m = masks[(module, param_type)]
            prune.custom_from_mask(module, param_type, m)

    @torch.no_grad()
    def compute_custom_masks(self, pruning_sparsity: float):
        from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount
        mask_dict = {}
        firstConvLayerIdx = [idx for idx, (module, param_type) in enumerate(self.parameters_to_prune) if
                             isinstance(module, torch.nn.Conv2d)][0]
        lastLinearLayerIdx = [idx for idx, (module, param_type) in enumerate(self.parameters_to_prune) if
                              isinstance(module, torch.nn.Linear)][-1]
        firstConv = getattr(*self.parameters_to_prune[firstConvLayerIdx])
        lastLinear = getattr(*self.parameters_to_prune[lastLinearLayerIdx])
        n_total_parameters = self.n_prunable_parameters
        k = float(_compute_nparams_toprune(pruning_sparsity, n_total_parameters))
        n_prunable_parameters_without_first = n_total_parameters - firstConv.numel()
        s_hat = k / n_prunable_parameters_without_first
        if s_hat > 1:
            raise ValueError(
                "Can't prune to desired sparsity while keeping first Conv layer dense.")
        elif s_hat > 0.8:
            s_prime = (k - 0.8 * lastLinear.numel()) / (n_total_parameters - firstConv.numel() - lastLinear.numel())
            if s_prime > 1:
                raise ValueError(
                    "Can't prune to desired sparsity while keeping first Conv layer dense and at least 20% of the last Linear layer non-pruned.")
            else:
                final_layer_pruning_sparsity = 0.8  # Set this to the maximum, i.e. 80%
                new_pruning_sparsity = s_prime  # Sparsity of middle layers
        elif s_hat <= 0.8:
            final_layer_pruning_sparsity = s_hat
            new_pruning_sparsity = s_hat

        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            if idx == firstConvLayerIdx:
                continue
            elif idx == lastLinearLayerIdx:
                layerSparsity = final_layer_pruning_sparsity
            else:
                layerSparsity = new_pruning_sparsity
            p = getattr(module, param_type)
            tensor_size = p.nelement()
            nparams_toprune = _compute_nparams_toprune(layerSparsity, tensor_size)
            _validate_pruning_amount(nparams_toprune, tensor_size)
            local_mask = torch.ones_like(p)
            if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
                # largest=True --> top k; largest=False --> bottom k
                # Prune the smallest k
                topk = torch.topk(
                    torch.abs(p).view(-1), k=nparams_toprune, largest=False
                )
                # topk will have .indices and .values
                local_mask.view(-1)[topk.indices] = 0
                mask_dict[(module, param_type)] = local_mask
        return mask_dict


class IMP_ERK(IMP_UniformPlus):
    """Erdős-Rényi-Kernel (ERK) as proposed in Evci et al. (2019).
        Adapted from https://github.com/jaeho-lee/layer-adaptive-sparsity"""

    def __init__(self, **kwargs):
        super(IMP_ERK, self).__init__(**kwargs)
        assert kwargs['n_phases'] == 1, 'Does not work with multiple phases at this point.'

    @torch.no_grad()
    def compute_custom_masks(self, pruning_sparsity: float):
        from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount
        mask_dict = {}
        erks = self.compute_erks()
        amounts = self.amounts_from_eps(ers=erks, amount=pruning_sparsity)

        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            p = getattr(module, param_type)
            tensor_size = p.nelement()
            pruning_amount = float(amounts[idx])
            nparams_toprune = _compute_nparams_toprune(pruning_amount, tensor_size)
            _validate_pruning_amount(nparams_toprune, tensor_size)
            local_mask = torch.ones_like(p)
            if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
                # largest=True --> top k; largest=False --> bottom k
                # Prune the smallest k
                topk = torch.topk(
                    torch.abs(p).view(-1), k=nparams_toprune, largest=False
                )
                # topk will have .indices and .values
                local_mask.view(-1)[topk.indices] = 0
                mask_dict[(module, param_type)] = local_mask
        return mask_dict

    @torch.no_grad()
    def amounts_from_eps(self, ers, amount):
        num_layers = ers.size(0)
        unmaskeds = torch.tensor([float(getattr(*param).numel()) for param in self.parameters_to_prune],
                                 device=ers.device)
        layers_to_keep_dense = torch.zeros(num_layers, device=ers.device)
        total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

        # Determine some layers to keep dense.
        is_eps_invalid = True
        while is_eps_invalid:
            to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

            ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
            survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

            layer_to_make_dense = -1
            max_ratio = 1.0
            for idx in range(num_layers):
                if layers_to_keep_dense[idx] == 0:
                    if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                        layer_to_make_dense = idx
                        max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

            if layer_to_make_dense == -1:
                is_eps_invalid = False
            else:
                layers_to_keep_dense[layer_to_make_dense] = 1

        amounts = torch.zeros(num_layers)

        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 1:
                amounts[idx] = 0.0
            else:
                amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
        return amounts

    @torch.no_grad()
    def compute_erks(self):
        erks = torch.zeros(len(self.parameters_to_prune), device=getattr(*self.parameters_to_prune[0]).device)
        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            p = getattr(module, param_type)
            if p.dim() == 4:
                erks[idx] = p.size(0) + p.size(1) + p.size(2) + p.size(3)
            else:
                erks[idx] = p.size(0) + p.size(1)
        return erks


#### Pruning stable strategies
class GSM(IMP):
    """Global Sparse Momentum as by Ding et al. 2019"""

    def __init__(self, desired_sparsity: float = None, n_epochs_per_phase: int = 1) -> None:
        assert desired_sparsity is not None, "Desired sparsity has not been provided."
        self.desired_sparsity = desired_sparsity
        super().__init__(desired_sparsity=desired_sparsity, n_epochs_per_phase=n_epochs_per_phase)
        self.Q = None

    def at_train_begin(self, model, LRScheduler):
        # Compute Q
        self.Q = int((1 - self.desired_sparsity) * self.n_prunable_parameters)

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()

    @torch.no_grad()
    def during_training(self, opt: torch.optim.Optimizer, **kwargs) -> None:
        """Apply topk mask to the gradients"""
        assert len(opt.param_groups) == 1, "This does not work for multiple param_groups yet."
        param_list = [p for group in opt.param_groups
                      for p in group['params'] if p.grad is not None]
        # Get the vector
        grad_vector = torch.cat([self.saliency_criterion(p=p).view(-1) for p in param_list])
        grad_vector_shape = grad_vector.shape
        device = param_list[0].device
        top_indices = torch.topk(grad_vector, k=self.Q).indices
        del grad_vector
        mask_vector = torch.zeros(grad_vector_shape, device=device)
        mask_vector[top_indices] = 1

        for p in param_list:
            numberOfElements = p.numel()
            partial_mask = mask_vector[:numberOfElements].view(p.shape)
            mask_vector = mask_vector[numberOfElements:]
            p.grad.mul_(partial_mask)  # Mask gradient

    def saliency_criterion(self, p):
        # Returns the saliency criterion for param p, i.e. torch.abs(p*p.grad)
        return torch.abs(p * p.grad)


class LC(IMP):
    """L0 Learning compression as in Carreira-Perpinan et al. (2018)"""

    def __init__(self, desired_sparsity: float = None, n_epochs_per_phase: int = 1, change_weight_decay_callback=None,
                 n_epochs_total=None, initial_weight_decay=None) -> None:
        assert desired_sparsity is not None, "Desired sparsity has not been provided."
        self.desired_sparsity = desired_sparsity
        self.change_weight_decay_callback = change_weight_decay_callback
        self.n_epochs_total = n_epochs_total
        self.initial_weight_decay = initial_weight_decay
        interval = 30
        if self.initial_weight_decay is not None and self.n_epochs_total is not None:
            self.penalty_per_time = {j * (self.n_epochs_total // interval): self.initial_weight_decay * (1.1 ** j) for j
                                     in range(interval)}
        else:
            self.penalty_per_time = {}

        super().__init__(desired_sparsity=desired_sparsity, n_epochs_per_phase=n_epochs_per_phase)

    @torch.no_grad()
    def during_training(self, opt: torch.optim.Optimizer, **kwargs) -> None:
        """Modify gradient such that only the n-k smallest weights are decayed"""
        assert len(opt.param_groups) == 1, "This does not work for multiple param_groups yet."
        group = opt.param_groups[0]
        weight_decay = group['weight_decay']
        if weight_decay == 0:
            # No need to do anything
            return
        param_list = [p for group in opt.param_groups
                      for p in group['params'] if p.grad is not None]
        # Get the vector
        param_vector = torch.cat([p.view(-1) for p in param_list])
        param_vector_shape = param_vector.shape
        device = param_list[0].device
        top_indices = torch.topk(torch.abs(param_vector), k=self.K).indices
        update_vector = torch.zeros(param_vector_shape, device=device)
        update_vector[top_indices] = param_vector[top_indices]
        del param_vector

        for p in param_list:
            numberOfElements = p.numel()
            partial_update_vector = update_vector[:numberOfElements].view(p.shape)
            update_vector = update_vector[numberOfElements:]
            p.grad.add_(partial_update_vector, alpha=-weight_decay)

    @torch.no_grad()
    def at_epoch_end(self, **kwargs):
        # Change weight decay
        epoch = kwargs['epoch']
        super().at_epoch_end(**kwargs)
        if epoch in self.penalty_per_time:
            self.change_weight_decay_callback(penalty=self.penalty_per_time[epoch])

    def finetuning_step(self, **kwargs):
        if self.change_weight_decay_callback is not None:
            # Disable weight decay at end of training
            self.change_weight_decay_callback(penalty=0.0)
        kwargs['finetuning_callback'](desired_sparsity=kwargs['desired_sparsity'],
                                      n_epochs_finetune=self.n_epochs_per_phase)

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()


class CS(Dense):
    """Continuous Sparsification as proposed by Savarese et al. (2019)."""

    def __init__(self, n_epochs_per_phase: int = 1, s_initial: float = 0, beta_final: float = 200, T_it=None) -> None:
        self.n_epochs_per_phase = n_epochs_per_phase
        self.s_initial = s_initial
        self.beta_final = beta_final
        self.T_it = T_it

        self.model = None
        self.original_parameters = dict()
        self.beta_current = 1
        self.sigma = torch.sigmoid
        self.scaling_factor = float(
            1. / self.sigma(torch.tensor(float(s_initial))))  # Rescale forward as is done in their implementation
        self.actual_sparsity = None
        super().__init__()

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.model = model
        super().after_initialization(model=model)
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
        torch.set_grad_enabled(
            enable_grad)  # This must come before the return if self.is_in_finetuning_phase, otherwise gradients are deactivated by the previous evaluation
        if self.is_in_finetuning_phase:
            return
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
        loss, lmbd = kwargs['loss'], kwargs['penalty']
        wd = kwargs['weight_decay']
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_mask") and hasattr(module, param_type + "_orig"):
                orig = getattr(module, param_type + "_orig")
                mask = getattr(module, param_type + "_mask")
                loss = loss + lmbd * torch.sum(
                    self.sigma(self.beta_current * mask))  # abs not needed since result is positive
                loss = loss + 0.5 * wd * torch.sum(orig ** 2)
        return loss

    @torch.no_grad()
    def after_training_iteration(self, it):
        """Called after each training iteration"""
        exponent = float(it) / self.T_it
        self.beta_current = self.beta_final ** exponent

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.actual_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        self.deregister_masks(opt=opt)  # This is equivalent to pruning
        self.actual_sparsity = metrics.metrics.global_sparsity(self.model)
        after_pruning_callback(desired_sparsity=self.actual_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.actual_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.actual_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)


class STR(Dense):
    """Soft Threshold Weight Reparameterization as proposed by Kusupati et al. (2020)."""

    def __init__(self, n_epochs_per_phase: int = 1, s_initial: float = 0, g_fn: str = 'sigmoid',
                 use_global_threshold: bool = False) -> None:
        self.n_epochs_per_phase = n_epochs_per_phase
        self.s_initial = s_initial
        self.use_global_threshold = use_global_threshold
        if g_fn == 'sigmoid':
            self.g = torch.sigmoid
        else:
            raise NotImplementedError(f"Threshold function {g_fn} not implemented.")

        self.model = None
        self.actual_sparsity = None
        super().__init__()

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.model = model
        super().after_initialization(model=model)
        self.register_masks()

    # Important: no @torch.no_grad()
    def apply_reparameterization(self, p, thresh):
        res = torch.sign(p) * torch.relu(torch.abs(p) - self.g(thresh))
        return res

    @torch.no_grad()
    def register_masks(self) -> None:
        """Add a learnable mask parameter for every param in self.parameters_to_prune"""
        self.model.thresh = None  # Needed even if local case for forward mode after pruning
        device = next(self.model.parameters()).device  # This wouldn't work if parameters lie on different devices
        if self.use_global_threshold:
            # Single Threshold
            self.model.thresh = torch.nn.Parameter(torch.tensor(float(self.s_initial), device=device),
                                                   requires_grad=True)

        for module, param_type in self.parameters_to_prune:
            orig = getattr(module, param_type)
            # Create thresh and _orig tensors
            if not self.use_global_threshold:
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
            if hasattr(module, param_type + "_thresh") or self.model.thresh is not None:
                thresh = getattr(module, param_type + "_thresh") if not self.use_global_threshold else self.model.thresh
                orig = getattr(module, param_type + "_orig")

                # Get sparse weights (Note: the final weights include the -threshold term for non-pruned weights
                # In other words: it is not safe to just prune using the thresholds
                final = self.apply_reparameterization(p=orig, thresh=thresh)  # Ensures that correct values are used
                binary_mask = torch.where(torch.abs(orig) > self.g(thresh), 1, 0)
                # Set original parameters before pruning
                setattr(module, param_type, torch.nn.Parameter(final))

                # Make .weight/.bias the Parameter, delete the rest
                module.register_parameter(param_type, getattr(module, param_type))
                if not self.use_global_threshold:
                    del module._parameters[param_type + "_thresh"]
                del module._parameters[param_type + "_orig"]

                # Prune from learned binary mask
                prune.custom_from_mask(module, param_type, binary_mask)

        # Delete thresh in the global case
        if self.use_global_threshold:
            del self.model._parameters['thresh']
            self.model.thresh = None

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
            if hasattr(module, param_type + "_thresh") or self.model.thresh is not None:
                thresh = getattr(module, param_type + "_thresh") if not self.use_global_threshold else self.model.thresh
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

    @torch.no_grad()
    def after_training_iteration(self, it):
        """Called after each training iteration"""
        pass

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.actual_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        self.deregister_masks(opt=opt)  # This is equivalent to pruning
        self.actual_sparsity = metrics.metrics.global_sparsity(self.model)
        after_pruning_callback(desired_sparsity=self.actual_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.actual_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.actual_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)

    @torch.no_grad()
    def get_per_layer_thresholds(self):
        """Returns a list of per layer thresholds"""
        thresholds = []
        if self.use_global_threshold and self.model.thresh is not None:
            thresholds.append(float(self.g(self.model.thresh)))
        elif not self.use_global_threshold:
            for module, param_type in self.parameters_to_prune:
                if hasattr(module, param_type + "_thresh"):
                    thresh = getattr(module, param_type + "_thresh")
                    thresholds.append(float(self.g(thresh)))
        return thresholds


class DST(Dense):
    """Dynamic Sparse Training as in Liu et al. (2020)"""

    def __init__(self, n_epochs_per_phase: int = 1) -> None:
        self.n_epochs_per_phase = n_epochs_per_phase

        self.model = None
        self.actual_sparsity = None

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
        super().__init__()

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.model = model
        super().after_initialization(model=model)
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
        loss, lmbd = kwargs['loss'], kwargs['penalty']
        wd = kwargs['weight_decay']
        for module, param_type in self.parameters_to_prune:
            if hasattr(module, param_type + "_thresh") and hasattr(module, param_type + "_orig"):
                orig = getattr(module, param_type + "_orig")
                thresh = getattr(module, param_type + "_thresh")
                loss = loss + lmbd * torch.sum(torch.exp(-thresh))
                loss = loss + 0.5 * wd * torch.sum(orig ** 2)
        return loss

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.actual_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        self.deregister_masks(opt=opt)  # This is equivalent to pruning
        self.actual_sparsity = metrics.metrics.global_sparsity(self.model)
        after_pruning_callback(desired_sparsity=self.actual_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.actual_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.actual_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)


class DNW(Dense):
    """Discovering Neural Wirings as in Wortsman et al. (2019)"""

    def __init__(self, desired_sparsity: float = None, n_epochs_per_phase: int = 1) -> None:
        self.desired_sparsity = desired_sparsity
        self.n_epochs_per_phase = n_epochs_per_phase

        self.model = None

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
        super().__init__()

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.model = model
        super().after_initialization(model=model)
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
            # Create _orig tensor

            # Make sure this is not a parameter of the model anymore
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(param_type + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[param_type]
            setattr(module, param_type, orig.detach().clone())

    @torch.no_grad()
    def deregister_masks(self, opt) -> None:
        with torch.no_grad():
            param_vector = torch.cat(
                [getattr(module, param_type + "_orig").view(-1) for module, param_type in self.parameters_to_prune
                 if hasattr(module, param_type + '_orig')])
            n_prune_global = int(self.desired_sparsity * self.n_prunable_parameters)  # Number of parameters to prune
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
        """Apply reparameterization in the forward and backward pass, no reverting needed except at the end of training"""
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
                    self.desired_sparsity * self.n_prunable_parameters)  # Number of parameters to prune
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

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.desired_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        self.deregister_masks(opt=opt)  # This is equivalent to pruning
        after_pruning_callback(desired_sparsity=self.desired_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.desired_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.desired_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)


class BIMP(Dense):
    """Budgeted IMP"""

    def __init__(self, model, n_train_epochs, n_train_budget, n_epochs_per_phase, desired_sparsity, pruning_interval,
                 after_pruning_callback):
        super().__init__()
        self.model = model
        self.n_train_epochs = n_train_epochs
        self.n_train_budget = n_train_budget
        self.n_epochs_per_phase = n_epochs_per_phase
        self.goal_sparsity = desired_sparsity
        self.pruning_interval = pruning_interval
        self.after_pruning_callback = after_pruning_callback

        assert self.n_train_budget <= self.n_train_epochs
        assert self.pruning_interval <= self.n_train_epochs - self.n_train_budget, "Pruning interval too large."

        self.current_sparsity = 0.0
        self.n_pruning_steps = (self.n_train_epochs - self.n_train_budget) // self.pruning_interval
        self.pruning_epochs = OrderedDict(
            {self.n_train_budget + int(round(t * self.pruning_interval)): 1 - (1 - self.goal_sparsity) ** (
                    float(t + 1) / self.n_pruning_steps) for t in
             range(self.n_pruning_steps)})
        self.currently_required_sparsity = self.current_sparsity

    def at_epoch_end(self, **kwargs):
        epoch, epoch_lr, train_loss, optimizer = kwargs['epoch'], kwargs['epoch_lr'], kwargs['train_loss'], kwargs[
            'optimizer']
        super().at_epoch_end(**kwargs)
        did_prune = self.pruning_scheduler(epoch=epoch)
        # No need to prune momentum, we prune hard anyway

    def pruning_scheduler(self, epoch):
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            current_density = 1 - self.current_sparsity
            sparsity_step = 1. - (1 - self.currently_required_sparsity) / current_density
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(self.model, pruning_sparsity=sparsity_step)
            return True
        return False

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.goal_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        # The functionality to retrain is implemented, although we technically retrain all the time
        # No pruning needed
        after_pruning_callback(desired_sparsity=self.goal_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.goal_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.goal_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)

    def get_pruning_method(self):
        return prune.L1Unstructured


class BIMP_LC(BIMP):
    """Budgeted IMP"""

    def pruning_scheduler(self, epoch):
        # Do the same but call the after pruning callback
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            current_density = 1 - self.current_sparsity
            sparsity_step = 1. - (1 - self.currently_required_sparsity) / current_density
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(self.model, pruning_sparsity=sparsity_step)
            self.after_pruning_callback(
                desired_sparsity=self.current_sparsity)
            return True
        return False


class BIMP_LCN(BIMP):
    """Budgeted IMP"""

    def __init__(self, scaling_callback, **kwargs):
        self.scaling_callback = scaling_callback
        super().__init__(**kwargs)

    def pruning_scheduler(self, epoch):
        # Do the same but compute the norm drop and don't call the after_pruning_callback
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            current_density = 1 - self.current_sparsity
            sparsity_step = 1. - (1 - self.currently_required_sparsity) / current_density
            self.current_sparsity = self.currently_required_sparsity
            is_last_pruning = (epoch == list(self.pruning_epochs.keys())[-1])
            self.scaling_callback(before_pruning=True, pruning_sparsity=sparsity_step, is_last_pruning=is_last_pruning)
            self.pruning_step(self.model, pruning_sparsity=sparsity_step)
            self.scaling_callback(before_pruning=False, pruning_sparsity=sparsity_step, is_last_pruning=is_last_pruning)
            return True
        return False


class GMP(Dense):
    """Gradual Magnitude Pruning as proposed by Zhu & Gupta (2017), but with a global parameter selection"""

    def __init__(self, model, n_train_epochs, n_epochs_per_phase, desired_sparsity, pruning_interval, allow_recovering,
                 after_pruning_callback):
        super().__init__()
        self.model = model
        self.n_train_epochs = n_train_epochs
        self.n_epochs_per_phase = n_epochs_per_phase
        self.goal_sparsity = desired_sparsity
        self.pruning_interval = pruning_interval
        self.allow_recovering = allow_recovering  # If True, then pruned weights can be come reactivated again
        self.after_pruning_callback = after_pruning_callback
        assert 2 * self.pruning_interval <= self.n_train_epochs, "Pruning interval too large."

        self.current_sparsity = 0.0
        self.n_pruning_steps = self.n_train_epochs // self.pruning_interval - 1
        self.pruning_epochs = OrderedDict(
            {int(round(t * self.pruning_interval)): self.sparsity_schedule(int(round(t * self.pruning_interval))) for t
             in
             range(1, self.n_pruning_steps + 1, 1)})
        self.currently_required_sparsity = self.current_sparsity

    def sparsity_schedule(self, t):
        return self.goal_sparsity + (0 - self.goal_sparsity) * (
                1 - t / int(round(self.pruning_interval * self.n_pruning_steps))) ** 3

    def at_epoch_end(self, **kwargs):
        epoch, epoch_lr, train_loss, optimizer = kwargs['epoch'], kwargs['epoch_lr'], kwargs['train_loss'], kwargs[
            'optimizer']
        super().at_epoch_end(**kwargs)
        did_prune = self.pruning_scheduler(epoch=epoch)
        if did_prune:
            # We need to prune momentum, otherwise the non-active weights are changed
            self.prune_momentum(optimizer=optimizer)

    def pruning_scheduler(self, epoch):
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            current_density = 1 - self.current_sparsity
            sparsity_step = 1. - (1 - self.currently_required_sparsity) / current_density
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(self.model, pruning_sparsity=sparsity_step, compute_from_scratch=self.allow_recovering)
            # if self.allow_recovering and epoch != list(self.pruning_epochs.keys())[-1]:
            # Note: We disabled the distinction since all methods now work also without allow_recovering
            self.current_sparsity = 0.0  # Adjust current sparsity, otherwise the next step doesnt reach the goal sparsity
            return True
        return False

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback(actual_sparsity=self.goal_sparsity)

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        # No pruning needed
        after_pruning_callback(desired_sparsity=self.goal_sparsity, reset_momentum=True)
        self.finetuning_step(desired_sparsity=self.goal_sparsity, finetuning_callback=finetuning_callback)
        save_model_callback(
            model_type=f"{self.goal_sparsity}-sparse_final")

    def finetuning_step(self, desired_sparsity, finetuning_callback):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase)

    def get_pruning_method(self):
        return prune.L1Unstructured


class GMP_Uniform(GMP):
    """GMP with a uniform selection criterion as proposed by Zhu & Gupta."""

    def __init__(self, **kwargs):
        # assert kwargs['allow_recovering'], "Allow recovering must be enabled for now."
        super(GMP_Uniform, self).__init__(**kwargs)

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):

        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the equivalence of weight_orig and weight
                orig = getattr(module, param_type + "_orig").detach().clone()
                prune.remove(module, param_type)
                if self.allow_recovering:
                    # We have to revert to weight_orig and then compute the mask
                    # Otherwise: # Make the mask permanent, then recompute the mask -> this is the same as hard pruning
                    p = getattr(module, param_type)
                    p.copy_(orig)
                del orig

        for module, param_type in self.parameters_to_prune:
            prune.l1_unstructured(module, name=param_type, amount=pruning_sparsity)


class GMP_UniformPlus(GMP_Uniform):
    """GMP as proposed by Zhu & Gupta, but with the selection criterion by Gale et al."""

    def __init__(self, **kwargs):
        super(GMP_UniformPlus, self).__init__(**kwargs)

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the equivalence of weight_orig and weight
                orig = getattr(module, param_type + "_orig").detach().clone()
                prune.remove(module, param_type)
                if self.allow_recovering:
                    # We have to revert to weight_orig and then compute the mask
                    # Otherwise: # Make the mask permanent, then recompute the mask -> this is the same as hard pruning
                    p = getattr(module, param_type)
                    p.copy_(orig)
                del orig

        masks = self.compute_custom_masks(pruning_sparsity=pruning_sparsity)
        for module, param_type in masks:
            m = masks[(module, param_type)]
            prune.custom_from_mask(module, param_type, m)

    @torch.no_grad()
    def compute_custom_masks(self, pruning_sparsity: float):
        from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount
        mask_dict = {}
        firstConvLayerIdx = [idx for idx, (module, param_type) in enumerate(self.parameters_to_prune) if
                             isinstance(module, torch.nn.Conv2d)][0]
        lastLinearLayerIdx = [idx for idx, (module, param_type) in enumerate(self.parameters_to_prune) if
                              isinstance(module, torch.nn.Linear)][-1]
        firstConv = getattr(*self.parameters_to_prune[firstConvLayerIdx])
        lastLinear = getattr(*self.parameters_to_prune[lastLinearLayerIdx])
        n_total_parameters = self.n_prunable_parameters
        k = float(_compute_nparams_toprune(pruning_sparsity, n_total_parameters))
        n_prunable_parameters_without_first = n_total_parameters - firstConv.numel()
        s_hat = k / n_prunable_parameters_without_first
        if s_hat > 1:
            raise ValueError(
                "Can't prune to desired sparsity while keeping first Conv layer dense.")
        elif s_hat > 0.8:
            s_prime = (k - 0.8 * lastLinear.numel()) / (n_total_parameters - firstConv.numel() - lastLinear.numel())
            if s_prime > 1:
                raise ValueError(
                    "Can't prune to desired sparsity while keeping first Conv layer dense and at least 20% of the last Linear layer non-pruned.")
            else:
                final_layer_pruning_sparsity = 0.8  # Set this to the maximum, i.e. 80%
                new_pruning_sparsity = s_prime  # Sparsity of middle layers
        elif s_hat <= 0.8:
            final_layer_pruning_sparsity = s_hat
            new_pruning_sparsity = s_hat

        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            if idx == firstConvLayerIdx:
                continue
            elif idx == lastLinearLayerIdx:
                layerSparsity = final_layer_pruning_sparsity
            else:
                layerSparsity = new_pruning_sparsity
            p = getattr(module, param_type)
            tensor_size = p.nelement()
            nparams_toprune = _compute_nparams_toprune(layerSparsity, tensor_size)
            _validate_pruning_amount(nparams_toprune, tensor_size)
            local_mask = torch.ones_like(p)
            if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
                # largest=True --> top k; largest=False --> bottom k
                # Prune the smallest k
                topk = torch.topk(
                    torch.abs(p).view(-1), k=nparams_toprune, largest=False
                )
                # topk will have .indices and .values
                local_mask.view(-1)[topk.indices] = 0
                mask_dict[(module, param_type)] = local_mask
        return mask_dict


class GMP_ERK(GMP_UniformPlus):
    """GMP with the ERK selection"""

    def __init__(self, **kwargs):
        super(GMP_ERK, self).__init__(**kwargs)

    @torch.no_grad()
    def compute_custom_masks(self, pruning_sparsity: float):
        from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount
        mask_dict = {}
        erks = self.compute_erks()
        amounts = self.amounts_from_eps(ers=erks, amount=pruning_sparsity)

        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            p = getattr(module, param_type)
            tensor_size = p.nelement()
            pruning_amount = float(amounts[idx])
            nparams_toprune = _compute_nparams_toprune(pruning_amount, tensor_size)
            _validate_pruning_amount(nparams_toprune, tensor_size)
            local_mask = torch.ones_like(p)
            if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
                # largest=True --> top k; largest=False --> bottom k
                # Prune the smallest k
                topk = torch.topk(
                    torch.abs(p).view(-1), k=nparams_toprune, largest=False
                )
                # topk will have .indices and .values
                local_mask.view(-1)[topk.indices] = 0
                mask_dict[(module, param_type)] = local_mask
        return mask_dict

    @torch.no_grad()
    def amounts_from_eps(self, ers, amount):
        num_layers = ers.size(0)
        unmaskeds = torch.tensor([float(getattr(*param).numel()) for param in self.parameters_to_prune],
                                 device=ers.device)
        layers_to_keep_dense = torch.zeros(num_layers, device=ers.device)
        total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

        # Determine some layers to keep dense.
        is_eps_invalid = True
        while is_eps_invalid:
            to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

            ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
            survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

            layer_to_make_dense = -1
            max_ratio = 1.0
            for idx in range(num_layers):
                if layers_to_keep_dense[idx] == 0:
                    if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                        layer_to_make_dense = idx
                        max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

            if layer_to_make_dense == -1:
                is_eps_invalid = False
            else:
                layers_to_keep_dense[layer_to_make_dense] = 1

        amounts = torch.zeros(num_layers)

        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 1:
                amounts[idx] = 0.0
            else:
                amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
        return amounts

    @torch.no_grad()
    def compute_erks(self):
        erks = torch.zeros(len(self.parameters_to_prune), device=getattr(*self.parameters_to_prune[0]).device)
        for idx, (module, param_type) in enumerate(self.parameters_to_prune):
            p = getattr(module, param_type)
            if p.dim() == 4:
                erks[idx] = p.size(0) + p.size(1) + p.size(2) + p.size(3)
            else:
                erks[idx] = p.size(0) + p.size(1)
        return erks


class GMP_LAMP(GMP):
    """GMP with LAMP selection"""

    def __init__(self, **kwargs):
        assert kwargs['allow_recovering'], "Allow recovering must be enabled for now."
        super(GMP_LAMP, self).__init__(**kwargs)

    def get_pruning_method(self):
        intermediate_pruning_class = lambda amount: LAMPUnstructured(parameters_to_prune=self.parameters_to_prune,
                                                                     amount=amount)
        return intermediate_pruning_class


class DPF(GMP):
    """Dynamic Pruning with Feedback as proposed by Lin et. al (2020)"""

    def __init__(self, model, n_train_epochs, n_epochs_per_phase, desired_sparsity, pruning_interval,
                 after_pruning_callback):
        allow_recovering = False
        self.original_parameters = dict()  # Saves the error between before_forward and after_forward
        super(DPF, self).__init__(model, n_train_epochs, n_epochs_per_phase, desired_sparsity, pruning_interval,
                                  allow_recovering, after_pruning_callback)

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
        # Pruning schedule is handled on iteration basis
        if epoch in self.pruning_epochs:
            # Prune
            self.currently_required_sparsity = self.pruning_epochs[epoch]
            self.current_sparsity = self.currently_required_sparsity
            self.pruning_step(self.model, pruning_sparsity=self.currently_required_sparsity, only_save_mask=True,
                              compute_from_scratch=True)
            return True
        return False

    @torch.no_grad()
    def after_training_iteration(self, it):
        """Called after each training iteration"""
        if it % 16 == 0 and not self.is_in_finetuning_phase:  # 16 is a hardcoded value used by the authors, i.e. every 16 iterations the mask update is triggered
            self.pruning_step(self.model, pruning_sparsity=self.currently_required_sparsity, only_save_mask=True,
                              compute_from_scratch=True)

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
