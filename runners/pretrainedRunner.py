# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         pretrainedRunner.py
# Description:  Runner class for starting from a pretrained model
# ===========================================================================
import os
import sys
import warnings

import numpy as np
import torch
import torch.optim as optimizers
import wandb

from runners.baseRunner import baseRunner
from utilities import Utilities as Utils


class pretrainedRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_run = None

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the model file."""
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state == 'failed':
                # Ignore this run
                continue
            self.trainedModelFile = run.summary.get('trained_model_file')
            try:
                if self.trainedModelFile is not None:
                    runsExist = True
                    run.file(self.trainedModelFile).download(root=wandb.run.dir)
                    self.seed = run.config['seed']
                    self.reference_run = run
                    break
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                self.trainedModelFile = None
        assert not (
                runsExist and self.trainedModelFile is None), "Runs found, but none of them have a model available -> abort."
        outputStr = f"Found {self.trainedModelFile} in run {run.name}" \
            if self.trainedModelFile is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference trained model in project: {outputStr}\n")
        assert self.trainedModelFile is not None, "No reference trained model found, Aborting."

    def get_missing_config(self):
        missing_config_keys = ['momentum',
                               'nesterov',
                               'nepochs',
                               'n_epochs_warmup']
        additional_dict = {
            'last_training_lr': self.reference_run.summary['trained.learning_rate'],
            'optimizer': 'SGD',
            'trained.test.accuracy': self.reference_run.summary['trained.test']['accuracy']
        }
        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = self.reference_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict['trained.test.accuracy']

    def define_optimizer_scheduler(self):
        # Define the optimizer using the parameters from the reference run
        self.optimizer = optimizers.SGD(params=self.model.parameters(), lr=self.config['last_training_lr'],
                                        momentum=self.config['momentum'],
                                        weight_decay=self.config['weight_decay'],
                                        nesterov=self.config['nesterov'])

    def fill_strategy_information(self):
        # Get the wandb information about lr's and losses and fill the corresponding strategy dicts, which can then be used by rewinders
        # Note: this only works if the reference model has "Dense" strategy
        for row in self.reference_run.history(keys=['learning_rate', 'train.loss', 'train.accuracy'], pandas=False):
            epoch, epoch_lr, train_loss, train_acc = row['_step'], row['learning_rate'], row['train.loss'], row['train.accuracy']
            self.strategy.at_epoch_end(epoch=epoch, epoch_lr=epoch_lr, train_loss=train_loss, train_acc=train_acc)

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        # Find the reference run
        filterDict = {"$and": [{"config.run_id": self.config.run_id},
                               {"config.model": self.config.model},
                               {"config.weight_decay": self.config.weight_decay}, {"config.strategy": "Dense"}]}
        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.learning_rate": self.config.learning_rate})

        self.find_existing_model(filterDict=filterDict)

        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True
        self.get_missing_config()  # Load keys that are missing in the config

        self.trainLoader, self.testLoader = self.get_dataloaders()
        self.model = self.get_model(load_initial=True)  # Load the trained model
        self.define_optimizer_scheduler()
        self.checkpoint_file = os.path.join(wandb.run.dir,
                                            self.trainedModelFile)  # Set the checkpoint file to the trainedModelFile
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

        # Define strategy
        self.strategy = self.define_strategy()
        self.strategy.after_initialization(model=self.model)  # To ensure that all parameters are properly set
        self.fill_strategy_information()
        # Run the computations
        self.strategy.at_train_end(model=self.model, finetuning_callback=self.fine_tuning,
                                   restore_callback=self.restore_model, save_model_callback=self.save_model,
                                   after_pruning_callback=self.after_pruning_callback, opt=self.optimizer)

        self.strategy.final(model=self.model, final_log_callback=self.final_log)
