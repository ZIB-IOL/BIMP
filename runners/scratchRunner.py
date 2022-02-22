# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         scratchRunner.py
# Description:  Runner class for methods that do *not* start from a pretrained model
# ===========================================================================
import os
import sys
import time


import numpy as np
import torch

import wandb

from utilities import Utilities as Utils
from runners.baseRunner import baseRunner


class scratchRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the model file."""
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False   # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state == 'failed':
                # Ignore this run
                continue
            try:
                self.trainedModelFile = run.summary.get('initial_model_file')
                seed = run.config.get('seed')
                if self.trainedModelFile is not None and seed is not None:
                    runsExist = True
                    run.file(self.trainedModelFile).download(root=wandb.run.dir)
                    self.seed = seed
                    break
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                self.trainedModelFile, seed = None, None
        assert not (runsExist and self.trainedModelFile is None), "Runs found, but none of them have a model available -> abort."
        outputStr = f"Found {self.trainedModelFile} with seed {seed}" if self.trainedModelFile is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference initial model in project: {outputStr}\n")

    def run(self):
        """Function controlling the workflow of scratchRunner"""
        if self.config.fixed_init:
            # If not existing, start a new model, otherwise use existing one with same run-id

            filterDict = {"$and": [{"config.run_id": self.config.run_id}, {"config.fixed_init": True},
                                   {"config.model": self.config.model}, ]}
            self.find_existing_model(filterDict=filterDict)

        if self.seed is None:
            # Generate a random seed
            self.seed = int((os.getpid() + 1) * time.time()) % 2 ** 32

        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True

        self.trainLoader, self.testLoader = self.get_dataloaders()
        self.model = self.get_model(load_initial=(self.trainedModelFile is not None))
        # Save initial model before training
        if self.config.fixed_init:
            if self.trainedModelFile is None:
                self.trainedModelFile = f"initial_model_run-{self.config.run_id}_seed-{self.seed}.pt"
                sys.stdout.write(f"Creating {self.trainedModelFile}.\n")
                self.save_model(model_type='initial')
            wandb.summary['initial_model_file'] = self.trainedModelFile
            wandb.save(self.trainedModelFile)
        self.strategy = self.define_strategy()
        self.strategy.after_initialization(model=self.model)
        self.define_optimizer_scheduler()

        self.strategy.at_train_begin(model=self.model, LRScheduler=self.scheduler)
        self.save_model(model_type='untrained')

        # Do initial prune if necessary
        self.strategy.initial_prune()

        # Do proper training
        self.train()
        # Save trained (unpruned) model
        self.checkpoint_file = self.save_model(model_type='trained')

        self.strategy.start_forward_mode()
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

        self.strategy.end_forward_mode()

        # Potentially finetune the final model
        self.strategy.at_train_end(model=self.model, finetuning_callback=self.fine_tuning,
                                   restore_callback=self.restore_model, save_model_callback=self.save_model,
                                   after_pruning_callback=self.after_pruning_callback, opt=self.optimizer)

        self.strategy.final(model=self.model, final_log_callback=self.final_log)
