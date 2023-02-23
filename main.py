# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         main.py
# Description:  Starts up a run for the comparison between sparsification strategies
# ===========================================================================

import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import torch
import wandb

from runners.pretrainedRunner import pretrainedRunner
from runners.scratchRunner import scratchRunner
from strategies import scratchStrategies

# Default wandb parameters
defaults = dict(
    # System
    run_id=None,
    computer=socket.gethostname(),
    # Setup
    dataset=None,
    arch=None,
    n_epochs=None,
    batch_size=None,
    # Efficiency
    use_amp=None,
    # Optimizer
    learning_rate=None,
    n_epochs_warmup=None,  # number of epochs to warm up the lr, should be an int
    momentum=None,
    weight_decay=None,
    # Sparsification strategy
    strategy=None,
    use_pretrained=None,
    goal_sparsity=None,
    # Retraining
    n_phases=None,  # Should be 1, except when using IMP
    n_epochs_per_phase=None,
    n_epochs_to_split=None,
    retrain_schedule=None,
    retrain_schedule_warmup=None,
    retrain_schedule_init=None,
    # GMP
    pruning_interval=None,
    allow_recovering=None,
    # STR
    s_initial=None,
    # DST
    penalty=None,
    # CS
    beta_final=None,
)

if '--debug' in sys.argv:
    defaults.update(dict(
        # System
        run_id=1,
        computer=socket.gethostname(),
        # Setup
        dataset='mnist',
        arch='Simple',
        n_epochs=5,
        batch_size=1028,
        # Efficiency
        use_amp=True,
        # Optimizer
        learning_rate='(Linear, 0.1)',
        n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
        momentum=0.9,
        weight_decay=0.0001,
        # Sparsification strategy
        strategy='DST',
        use_pretrained=None,
        goal_sparsity=0.8,
        # Retraining
        n_phases=2,  # Should be 1, except when using IMP
        n_epochs_per_phase=1,
        n_epochs_to_split=None,
        retrain_schedule='ALLR',
        retrain_schedule_warmup=None,
        retrain_schedule_init=None,
        # GMP
        pruning_interval=1,
        allow_recovering=True,
        # STR
        s_initial=-0.1,
        # DST
        penalty=0.002,
        # CS
        beta_final=300,
    ))

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
n_gpus = torch.cuda.device_count()
if n_gpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


@contextmanager
def tempdir():
    """Create a temporary directory that is automatically removed after use."""
    tmp_root = '/tmp'
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # At the moment, IMP is the only strategy that requires a pretrained model, all others start from scratch
    config.update({'tmp_dir': tmp_dir})
    if config.use_pretrained is not None:
        # Use the pretrainedRunner
        runner = pretrainedRunner(config=config)
    else:
        # Use the scratchRunner
        try:
            check_for_strategy_existence = getattr(scratchStrategies, config.strategy)
        except Exception as e:
            raise NotImplementedError("Strategy does not exist, potentially forgot to specify 'use_pretrained'.")
        runner = scratchRunner(config=config)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
