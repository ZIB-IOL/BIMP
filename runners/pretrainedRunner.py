# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         pretrainedRunner.py
# Description:  Runner class for starting from a pretrained model
# ===========================================================================
import json
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch
import wandb

from runners.baseRunner import baseRunner
from strategies import scratchStrategies
from utilities.utilities import Utilities as Utils


class pretrainedRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_run = None

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the model file."""
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        # Some variables have to be extracted and checked manually, e.g. weight decay in scientific format
        manualVariables = ['weight_decay', 'penalty', 'group_penalty']
        manVarDict = {}
        dropIndices = []
        for var in manualVariables:
            for i in range(len(filterDict['$and'])):
                entry = filterDict['$and'][i]
                s = f"config.{var}"
                if s in entry:
                    dropIndices.append(i)
                    manVarDict[var] = entry[s]
        for idx in reversed(sorted(dropIndices)): filterDict['$and'].pop(idx)

        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state == 'failed':
                # Ignore this run
                continue
            # Check if run satisfies the manual variables
            conflict = False
            for var, val in manVarDict.items():
                if var in run.config and run.config[var] != val:
                    conflict = True
                    break
            if conflict:
                continue

            self.checkpoint_file = run.summary.get('trained_model_file')
            try:
                if self.checkpoint_file is not None:
                    runsExist = True
                    run.file(self.checkpoint_file).download(root=self.tmp_dir)
                    self.seed = run.config['seed']
                    self.reference_run = run
                    break
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                self.checkpoint_file = None
        assert not (
                runsExist and self.checkpoint_file is None), "Runs found, none have a model available -> abort."
        outputStr = f"Found {self.checkpoint_file} in run {run.name}" \
            if self.checkpoint_file is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference trained model in project: {outputStr}\n")
        assert self.checkpoint_file is not None, "No reference trained model found, Aborting."

    def get_missing_config(self):
        missing_config_keys = ['momentum',
                               'n_epochs_warmup',
                               'n_epochs']  # Have to have n_epochs, otherwise ALLR doesn't have this
        additional_dict = {
            'last_training_lr': self.reference_run.summary['trained.learning_rate'],
            'trained.test.accuracy': self.reference_run.summary['trained.test']['accuracy'],
            'trained.train.accuracy': self.reference_run.summary['trained.train']['accuracy'],
            'trained.train.loss': self.reference_run.summary['trained.train']['loss'],
        }
        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = self.reference_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict['trained.test.accuracy']
        self.trained_train_loss = additional_dict['trained.train.loss']
        self.trained_train_accuracy = additional_dict['trained.train.accuracy']

    def define_optimizer_scheduler(self):
        # Define the optimizer using the parameters from the reference run
        wd = self.config['weight_decay'] or 0.
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.config['last_training_lr'],
                                         momentum=self.config['momentum'],
                                         weight_decay=wd,
                                         nesterov=wd > 0.)

    def fill_strategy_information(self):
        # Get the wandb information about the learning rate, which can then be used by retraining schedules
        f = self.reference_run.file('iteration-lr-dict.json').download(root=self.tmp_dir)
        with open(f.name) as json_file:
            loaded_dict = json.load(json_file)
            self.strategy.lr_dict = OrderedDict(loaded_dict)

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        # Find the reference run
        filterDict = {"$and": [{"config.run_id": self.config.run_id},
                               {"config.arch": self.config.arch},
                               {"config.optimizer": self.config.optimizer},
                               ]}

        print(f"Searching for pretrained model of strategy: {self.config.use_pretrained}")
        filterDict["$and"].append({"config.strategy": self.config.use_pretrained})

        # Pull required parameters from scratchStrategies
        required_params = getattr(scratchStrategies, self.config.use_pretrained).required_params
        print(f"Requires parameters:", required_params)

        for hparam in required_params:
            if self.config[hparam] is None:
                sys.stdout.write(
                    f"Parameter {hparam} is required for strategy {self.config.use_pretrained} but not specified.\n")
            else:
                filterDict["$and"].append({f"config.{hparam}": self.config[hparam]})

        attributeList = ['weight_decay']

        for attr in attributeList:
            name, val = f"config.{attr}", self.config[attr]
            filterDict["$and"].append({name: val})

        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection "
                "of the pretrained model.")
            filterDict["$and"].append({"config.learning_rate": self.config.learning_rate})
        if self.config.n_epochs is not None:
            warnings.warn(
                "You specified n_epochs for retraining. Note that this only controls the selection of the pretrained "
                "model.")
            filterDict["$and"].append({"config.n_epochs": self.config.n_epochs})

        self.find_existing_model(filterDict=filterDict)
        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        torch.backends.cudnn.benchmark = True
        self.get_missing_config()  # Load keys that are missing in the config

        self.trainLoader, self.valLoader, self.testLoader = self.get_dataloaders()
        self.model = self.get_model(reinit=True, temporary=True)  # Load the trained model

        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)
        # Define strategy
        self.strategy = self.define_strategy()
        self.strategy.set_to_finetuning_phase()
        self.strategy.after_initialization()  # To ensure that all parameters are properly set
        self.define_optimizer_scheduler()  # This has to be after the definition of the strategy.
        self.strategy.set_optimizer(opt=self.optimizer)
        self.fill_strategy_information()

        # Run the computations
        self.strategy.at_train_end()

        self.strategy.final()
