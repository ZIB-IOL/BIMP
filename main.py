# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         main.py
# Description:  Starts up a run for the comparison between sparsification strategies
# ===========================================================================

import socket
import sys
import os
import shutil
import torch
import wandb

from runners.scratchRunner import scratchRunner
from runners.pretrainedRunner import pretrainedRunner

# Default wandb parameters
defaults = dict(
    # System
    run_id=None,
    computer=socket.gethostname(),
    fixed_init=None,
    # Setup
    dataset=None,
    model=None,
    nepochs=None,
    batch_size=None,
    # Optimizer (SGD)
    learning_rate=None,
    n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
    momentum=None,
    nesterov=None,
    weight_decay=None,
    # Sparsifying strategy
    strategy=None,
    goal_sparsity=None,
    # IMP
    IMP_selector=None,  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
    # Retraining
    n_phases=None,  # Should be 1, except when using IMP
    n_epochs_per_phase=None,
    retrain_schedule=None,
    retrain_schedule_warmup=None,
    retrain_schedule_init=None,
    dynamic_retrain_length=None,
    # BIMP
    n_train_budget=None,
    # GMP
    pruning_interval=None,
    allow_recovering=None,
    GMP_selector=None,  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
    # CS/STR/DST
    lmbd=None,  # Different from weight decay
    s_initial=None,
    beta_final=None,
    use_global_threshold=None,
)
debug_mode = False
if '--debug' in sys.argv:
    debug_mode = True
    defaults.update(dict(
        # System
        run_id=1,
        computer=socket.gethostname(),
        fixed_init=False,
        # Setup
        dataset='mnist',
        model='Simple',
        nepochs=4,
        batch_size=1028,
        # Optimizer (SGD)
        learning_rate='(MultiStepLR, 0.1, [3|7], 0.1)',
        # learning_rate='(CyclicBudgetLC, 0.1)',
        n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0005,
        # Sparsifying strategy
        strategy='IMP',
        goal_sparsity=0.99,
        # IMP
        IMP_selector='global',  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
        # Retraining
        n_phases=1,  # Should be 1, except when using IMP
        n_epochs_per_phase=2,
        retrain_schedule='LLR',
        retrain_schedule_warmup=None,
        retrain_schedule_init=None,
        dynamic_retrain_length=None,
        # BIMP
        n_train_budget=2,
        # GMP
        pruning_interval=1,
        allow_recovering=False,
        GMP_selector='uniform',  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
        # CS/STR/DST
        lmbd=1e-4,  # Different from weight decay
        s_initial=-0.1,
        beta_final=200,
        use_global_threshold=False,
    ))

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
ngpus = torch.cuda.device_count()
if ngpus > 0:
    if ngpus > 1 and config.dataset == 'imagenet':
        config.update(dict(device='cuda:' + ','.join(f"{i}" for i in range(ngpus))))
    else:
        config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))

# At the moment, IMP is the only strategy that requires a pretrained model, all others start from scratch
if config.strategy == 'IMP':
    # Use the pretrainedRunner
    runner = pretrainedRunner(config=config, debug_mode=debug_mode)
else:
    # Use the scratchRunner
    runner = scratchRunner(config=config, debug_mode=debug_mode)
runner.run()

# Close wandb run
wandb_dir_path = wandb.run.dir
wandb.join()

# Delete the local files
if os.path.exists(wandb_dir_path):
    shutil.rmtree(wandb_dir_path)
