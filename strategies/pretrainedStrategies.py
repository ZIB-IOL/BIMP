# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         strategies/pretrainedStrategies.py
# Description:  Sparsification strategies for pretrained models
# ===========================================================================
import torch.nn.utils.prune as prune

from strategies import scratchStrategies


#### Base Class
class IMP(scratchStrategies.Dense):
    """Iterative Magnitude Pruning Base Class"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_phases = self.run_config['n_phases']
        self.n_epochs_per_phase = self.run_config['n_epochs_per_phase']
        self.n_epochs_to_split = self.run_config['n_epochs_to_split']

        if self.n_epochs_to_split is not None:
            assert self.n_epochs_per_phase in [None, 0]
            if self.n_epochs_to_split % self.n_phases == 0:
                self.n_epochs_per_phase = {p: self.n_epochs_to_split // self.n_phases for p in
                                           range(1, self.n_phases + 1, 1)}
            else:
                self.n_epochs_per_phase = {p: self.n_epochs_to_split // self.n_phases for p in
                                           range(1, self.n_phases, 1)}
                self.n_epochs_per_phase[self.n_phases] = self.n_epochs_to_split - (self.n_phases - 1) * (
                        self.n_epochs_to_split // self.n_phases)
        else:
            self.n_epochs_per_phase = {p: self.n_epochs_per_phase for p in range(1, self.n_phases + 1, 1)}

    def at_train_end(self, **kwargs):
        # Sparsity factor on remaining weights after each round, yields desired_sparsity after all rounds
        prune_per_phase = 1 - (1 - self.goal_sparsity) ** (1. / self.n_phases)
        for phase in range(1, self.n_phases + 1, 1):
            self.pruning_step(pruning_sparsity=prune_per_phase)
            self.current_sparsity = 1 - (1 - prune_per_phase) ** phase
            self.callbacks['after_pruning_callback']()
            self.finetuning_step(pruning_sparsity=prune_per_phase, phase=phase)

    def finetuning_step(self, pruning_sparsity, phase):
        self.callbacks['finetuning_callback'](pruning_sparsity=pruning_sparsity,
                                              n_epochs_finetune=self.n_epochs_per_phase[phase],
                                              phase=phase)

    def get_pruning_method(self):
        return prune.L1Unstructured

    def final(self):
        super().final()
        self.callbacks['final_log_callback']()
