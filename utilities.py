# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         utilities.py
# Description:  Contains a variety of useful functions.
# ===========================================================================

import torch
import warnings
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount, _validate_pruning_amount_init
from bisect import bisect_right
import math


class Utilities:

    @staticmethod
    @torch.no_grad()
    def categorical_accuracy(y_true, output, topk=1):
        """Computes the precision@k for the specified values of k"""
        prediction = output.topk(topk, dim=1, largest=True, sorted=False).indices.t()
        n_labels = float(len(y_true))
        return prediction.eq(y_true.expand_as(prediction)).sum().item() / n_labels

    @staticmethod
    @torch.no_grad()
    def get_layer_norms(model):
        """Computes L1, L2, Linf norms of all layers of model"""
        layer_norms = dict()
        for name, param in model.named_parameters():
            idxLastDot = name.rfind(".")
            layer_name, weight_type = name[:idxLastDot], name[idxLastDot + 1:]
            if layer_name not in layer_norms:
                layer_norms[layer_name] = dict()
            layer_norms[layer_name][weight_type] = dict(
                L1=float(torch.norm(param, p=1)),
                L2=float(torch.norm(param, p=2)),
                Linf=float(torch.norm(param, p=float('inf')))
            )

        return layer_norms

    @staticmethod
    @torch.no_grad()
    def get_model_norm_square(model):
        """Get L2 norm squared of parameter vector. This works for a pruned model as well."""
        squared_norm = 0.
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    param = getattr(module, param_type)
                    squared_norm += torch.norm(param, p=2) ** 2
        return float(squared_norm)


class FixedLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Just uses the learning rate given by a list
    """

    def __init__(self, optimizer, lrList, last_epoch=-1):
        self.lrList = lrList

        super(FixedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.lrList[self.last_epoch] for group in self.optimizer.param_groups]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def result(self):
        return self.avg

    def __call__(self, val, n=1):
        """val is an average over n samples. To compute the overall average, add val*n to sum and increase count by n"""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


class LAMPUnstructured(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the appropriate LAMP-Score.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, parameters_to_prune, amount):
        # Check range of validity of pruning amount
        self.parameters_to_prune = parameters_to_prune  # This is a (non-sliced) vector that is passed implicitly
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # In the global case, t is already the global parameter vector, same for the mask
        # In the multiple pruning case, we only get the slice, hence we have to do LAMP as if t was all parameters as a vector
        # BUT: For LAMP it is important to distinguish layers

        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        tensor_list = []
        length_done = 0
        # Modified from https://github.com/jaeho-lee/layer-adaptive-sparsity
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                p_mask = getattr(module, param_type + '_mask')
                mask_length = int((p_mask == 1).sum())  # Get the number of entries that are still pruneable
            else:
                p_base = getattr(module, param_type)
                mask_length = int(p_base.numel())
            p = t[length_done:length_done + mask_length]
            assert p.numel() == mask_length
            length_done += mask_length

            sorted_scores, sorted_indices = torch.sort(torch.pow(p.flatten(), 2),
                                                       descending=False)  # Get indices to ascending sort
            scores_cumsum_temp = sorted_scores.cumsum(dim=0)
            scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=p.device)
            scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]

            # normalize by cumulative sum
            sorted_scores /= (sorted_scores.sum() - scores_cumsum)
            # tidy up and output
            final_scores = torch.zeros(scores_cumsum.shape, device=p.device)
            final_scores[sorted_indices] = sorted_scores
            tensor_list.append(final_scores)
        score_tensor = torch.cat(tensor_list)
        assert score_tensor.numel() == t.numel()
        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(
                score_tensor.view(-1), k=nparams_toprune, largest=False
            )
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(LAMPUnstructured, cls).apply(module, name, amount=amount)


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()


class ChainedSchedulers(torch.optim.lr_scheduler.ChainedScheduler):
    """
    Repairs ChainedScheduler to avoid a known bug that makes it into the pytorch release soon
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(ChainedSchedulers, self).__init__(**kwargs)


class CyclicLRAdaptiveBase(torch.optim.lr_scheduler.CyclicLR):

    def __init__(self, base_lr_scale_fn=None, **kwargs):
        self.base_lr_scale_fn = base_lr_scale_fn
        super(CyclicLRAdaptiveBase, self).__init__(**kwargs)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        # Adjust the base lrs
        if self.base_lr_scale_fn:
            for entry_idx in range(len(self.base_lrs)):
                self.base_lrs[entry_idx] = self.max_lrs[entry_idx] * self.base_lr_scale_fn(cycle)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs
