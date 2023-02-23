# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         baseRunner.py
# Description:  Base Runner class, all other runners inherit from this one
# ===========================================================================
import importlib
import os
import sys
import time
from collections import OrderedDict
from math import sqrt

import numpy as np
import torch
import wandb
from barbar import Bar
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy as Accuracy

from config import datasetDict, trainTransformDict, testTransformDict
from metrics import metrics
from strategies import scratchStrategies, pretrainedStrategies
from utilities.lr_schedulers import SequentialSchedulers, FixedLR
from utilities.utilities import Utilities as Utils


class baseRunner:
    """Base class for all runners, defines the general functions."""

    def __init__(self, config):
        self.config = config
        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)

        # Set a couple useful variables
        self.checkpoint_file = None
        self.trained_test_accuracy = None
        self.trained_train_loss = None
        self.trained_train_accuracy = None
        self.after_pruning_metrics = None
        self.seed = None
        self.squared_model_norm = None
        self.n_warmup_epochs = None
        self.trainIterationCtr = 1
        self.tmp_dir = config['tmp_dir']
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")
        self.ampGradScaler = None
        self.num_workers = None
        self.ultimate_log_dict = None

        # Variables to be set by inheriting classes
        self.strategy = None
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
        self.n_datapoints = None
        self.model = None
        self.trainData = None
        self.n_total_iterations = None

        if self.config.dataset in ['mnist', 'cifar10']:
            self.n_classes = 10
        elif self.config.dataset in ['cifar100']:
            self.n_classes = 100
        elif self.config.dataset in ['imagenet']:
            self.n_classes = 1000
        else:
            raise NotImplementedError

        # Define the loss object and metrics
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device=self.device)

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'accuracy': Accuracy(num_classes=self.n_classes).to(device=self.device),
                               'ips_throughput': MeanMetric().to(device=self.device)}
                        for mode in ['train', 'val', 'test']}

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self):
        """Collects and returns the metrics as a dictionary"""
        with torch.no_grad():
            self.strategy.start_forward_mode()  # Necessary to have correct computations for DPF
            n_total, n_nonzero = metrics.get_parameter_count(model=self.model)

            x_input, y_target = next(iter(self.valLoader))
            x_input, y_target = x_input.to(self.device), y_target.to(self.device)  # Move to CUDA if possible
            n_flops, n_nonzero_flops = metrics.get_flops(model=self.model, x_input=x_input)

            distance_to_pruned, rel_distance_to_pruned = {}, {}
            if self.config.goal_sparsity is not None:
                distance_to_pruned, rel_distance_to_pruned = metrics.get_distance_to_pruned(model=self.model,
                                                                                            sparsity=self.config.goal_sparsity)

            loggingDict = dict(
                train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items() if
                       getattr(metric, 'mode', True) is not None},  # Check if metric computable
                val={metric_name: metric.compute() for metric_name, metric in self.metrics['val'].items()},
                global_sparsity=metrics.global_sparsity(module=self.model),
                modular_sparsity=metrics.modular_sparsity(parameters_to_prune=self.strategy.parameters_to_prune),
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                nonzero_inference_flops=n_nonzero_flops,
                baseline_inference_flops=n_flops,
                theoretical_speedup=metrics.get_theoretical_speedup(n_flops=n_flops, n_nonzero_flops=n_nonzero_flops),
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                distance_to_origin=metrics.get_distance_to_origin(self.model),
                distance_to_pruned=distance_to_pruned,
                rel_distance_to_pruned=rel_distance_to_pruned,
            )

            loggingDict['test'] = dict()
            for metric_name, metric in self.metrics['test'].items():
                try:
                    # Catch case where MeanMetric mode not set yet
                    loggingDict['test'][metric_name] = metric.compute()
                except Exception:
                    continue

            self.strategy.end_forward_mode()  # Necessary to have correct computations for DPF
        return loggingDict

    def get_dataloaders(self):
        """Returns the dataloaders for the current dataset"""
        rootPath = f"./datasets_pytorch/{self.config.dataset}"

        if self.config.dataset == 'imagenet':
            trainData = datasetDict[self.config.dataset](root=rootPath, split='train',
                                                         transform=trainTransformDict[
                                                             self.config.dataset])
            testData = datasetDict[self.config.dataset](root=rootPath, split='val', transform=testTransformDict[
                self.config.dataset])
        else:
            trainData = datasetDict[self.config.dataset](root=rootPath, train=True,
                                                         download=True,
                                                         transform=trainTransformDict[
                                                             self.config.dataset])

            testData = datasetDict[self.config.dataset](root=rootPath, train=False,
                                                        transform=testTransformDict[
                                                            self.config.dataset])
        train_size = int(0.9 * len(trainData))
        val_size = len(trainData) - train_size
        self.trainData, valData = torch.utils.data.random_split(trainData, [train_size, val_size],
                                                                generator=torch.Generator().manual_seed(42))
        self.n_datapoints = train_size

        if self.config.dataset in ['imagenet', 'cifar100']:
            self.num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            self.num_workers = 2 if torch.cuda.is_available() else 0

        trainLoader = torch.utils.data.DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                                  pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)
        testLoader = torch.utils.data.DataLoader(testData, batch_size=self.config.batch_size, shuffle=False,
                                                 pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)

        return trainLoader, valLoader, testLoader

    def get_model(self, reinit: bool, temporary: bool = True):
        """Returns the model."""
        if reinit:
            # Define the model
            model = getattr(importlib.import_module('models.' + self.config.dataset), self.config.arch)()
        else:
            # The model has been initialized already
            model = self.model
            # Note, we have to get rid of all existing prunings, otherwise we cannot load the state_dict as it is unpruned
            if self.strategy:
                self.strategy.make_pruning_permanent(model=self.model)

        file = self.checkpoint_file
        if file is not None:
            dir = wandb.run.dir if not temporary else self.tmp_dir
            fPath = os.path.join(dir, file)

            state_dict = torch.load(fPath, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                v_new = v  # Remains unchanged if not in _orig format
                if k.endswith("_orig"):
                    # We loaded the _orig tensor and corresponding mask
                    name = name[:-5]  # Truncate the "_orig"
                    if f"{k[:-5]}_mask" in state_dict.keys():
                        v_new = v * state_dict[f"{k[:-5]}_mask"]

                new_state_dict[name] = v_new

            maskKeys = [k for k in new_state_dict.keys() if k.endswith("_mask")]
            for k in maskKeys:
                del new_state_dict[k]

            # Load the state_dict
            model.load_state_dict(new_state_dict)

        if self.dataParallel and reinit and not isinstance(model,
                                                           torch.nn.DataParallel):  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def define_optimizer_scheduler(self):
        """Defines the optimizer and the learning rate scheduler."""
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = self.config.learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
            scheduler = (name, kwargs)
            self.initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        wd = self.config['weight_decay'] or 0.
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.initial_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=wd, nesterov=wd > 0.)

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = len(self.trainLoader)
        n_total_iterations = iterations_per_epoch * self.config.n_epochs
        self.n_total_iterations = n_total_iterations
        n_warmup_iterations = 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = self.initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if self.config.n_epochs_warmup and self.config.n_epochs_warmup > 0:
            assert int(
                self.config.n_epochs_warmup) == self.config.n_epochs_warmup, "At the moment no float warmup allowed."
            n_warmup_iterations = int(float(self.config.n_epochs_warmup) * iterations_per_epoch)
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        name, kwargs = scheduler
        scheduler = None
        if name == 'Constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif name == 'StepLR':
            # Tuple of form ('StepLR', initial_lr, step_size, gamma)
            # Reduces initial_lr by gamma every step_size epochs
            step_size, gamma = int(kwargs[1]), float(kwargs[2])

            # Convert to iterations
            step_size = iterations_per_epoch * step_size

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size,
                                                        gamma=gamma)
        elif name == 'MultiStepLR':
            # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
            # Reduces initial_lr by gamma every epoch that is in the list milestones
            milestones, gamma = kwargs[1].strip(), float(kwargs[2])
            # Remove square bracket
            if milestones[0] == '[':
                milestones = milestones[1:]
            if milestones[-1] == ']':
                milestones = milestones[:-1]
            # Convert to iterations directly
            milestones = [int(ms) * iterations_per_epoch for ms in milestones.split('|')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                             gamma=gamma)
        elif name == 'ExponentialLR':
            # Tuple of form ('ExponentialLR', initial_lr, gamma)
            gamma = float(kwargs[1])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif name == 'Linear':
            if len(kwargs) == 2:
                # The final learning rate has also been passed
                end_factor = float(kwargs[1]) / float(kwargs[0])
            else:
                end_factor = 0.
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=end_factor,
                                                          total_iters=n_remaining_iterations)
        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        # Reset base lrs to make this work
        scheduler.base_lrs = [self.initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        elif name in ['StepLR', 'MultiStepLR']:
            # We need parallel schedulers, since the steps should be counted during warmup
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def define_strategy(self):
        """Define the training strategy."""
        # Define callbacks
        callbackDict = {
            'after_pruning_callback': self.after_pruning_callback,
            'finetuning_callback': self.fine_tuning,
            'restore_callback': self.restore_model,
            'save_model_callback': self.save_model,
            'final_log_callback': self.final_log,
        }
        # Base strategies
        if self.config.use_pretrained is not None:
            # Use pretrained model
            return getattr(pretrainedStrategies, self.config.strategy)(model=self.model, n_classes=self.n_classes,
                                                                       config=self.config, callbacks=callbackDict)
        else:
            # Start from scratch
            return getattr(scratchStrategies, self.config.strategy)(model=self.model, n_classes=self.n_classes,
                                                                    config=self.config, callbacks=callbackDict)

    def log(self, runTime, finetuning: bool = False, final_logging: bool = False):
        """Log to wandb."""
        loggingDict = self.get_metrics()
        self.strategy.start_forward_mode()
        loggingDict.update({'epoch_run_time': runTime})
        if not finetuning:
            # Update final trained metrics (necessary to be able to filter via wandb)
            for metric_type, val in loggingDict.items():
                wandb.run.summary[f"trained.{metric_type}"] = val
            # The usual logging of one epoch
            wandb.log(
                loggingDict
            )

        else:
            if not final_logging:
                wandb.log(
                    dict(finetune=loggingDict,
                         ),
                )
            else:
                # We add the after_pruning_metrics and don't commit, since the values are updated by self.final_log
                self.ultimate_log_dict = dict(finetune=loggingDict,
                                              pruned=self.after_pruning_metrics,
                                              )
        self.strategy.end_forward_mode()

    def final_log(self):
        """This function should only be called by pretrained strategies using the final sparsified model."""
        # Recompute accuracy and loss
        sys.stdout.write(
            f"\nFinal logging\n")
        self.reset_averaged_metrics()

        self.evaluate_model(data='val')
        self.evaluate_model(data='test')

        # Update final trained metrics (necessary to be able to filter via wandb)
        loggingDict = self.get_metrics()
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"final.{metric_type}"] = val

        # Update after prune metrics
        for metric_type, val in self.after_pruning_metrics.items():
            wandb.run.summary[f"pruned.{metric_type}"] = val

        # Add to existing self.ultimate_log_dict which was not committed yet
        if self.ultimate_log_dict is not None:
            if loggingDict['train']['accuracy'] == 0:
                del loggingDict['train']

            self.ultimate_log_dict['finetune'].update(loggingDict)
        else:
            self.ultimate_log_dict = {'finetune': loggingDict}

        wandb.log(self.ultimate_log_dict)
        Utils.dump_dict_to_json_wandb(metrics.per_layer_sparsity(model=self.model), 'sparsity_distribution')

    def after_pruning_callback(self):
        """Collects pruning metrics. Is called ONCE per run, namely on the LAST PRUNING step."""
        # Make the pruning permanent (this is in conflict with strategies that do not have a permanent pruning)
        self.strategy.enforce_prunedness()

        # Compute losses, accuracies after pruning
        sys.stdout.write(f"\nGoal sparsity reached - Computing incurred losses after pruning.\n")
        self.reset_averaged_metrics()

        self.evaluate_model(data='train')
        self.evaluate_model(data='val')
        self.evaluate_model(data='test')
        if self.squared_model_norm is not None:
            L2_norm_square = Utils.get_model_norm_square(self.model)
            norm_drop = sqrt(abs(self.squared_model_norm - L2_norm_square))
            if float(sqrt(self.squared_model_norm)) > 0:
                relative_norm_drop = norm_drop / float(sqrt(self.squared_model_norm))
            else:
                relative_norm_drop = {}
        else:
            norm_drop, relative_norm_drop = {}, {}

        pruning_instability, pruning_stability = {}, {}
        train_loss_increase, relative_train_loss_increase_factor = {}, {}
        if self.trained_test_accuracy is not None and self.trained_test_accuracy > 0:
            pruning_instability = (self.trained_test_accuracy - self.metrics['test']['accuracy'].compute()) \
                                  / self.trained_test_accuracy
            pruning_stability = 1 - pruning_instability
        if self.trained_train_loss is not None and isinstance(self.trained_train_loss,
                                                              float) and self.trained_train_loss > 0:
            train_loss_increase = self.metrics['train']['loss'].compute() - self.trained_train_loss
            relative_train_loss_increase_factor = train_loss_increase / self.trained_train_loss

        self.after_pruning_metrics = dict(
            train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items()},
            val={metric_name: metric.compute() for metric_name, metric in self.metrics['val'].items()},
            test={metric_name: metric.compute() for metric_name, metric in self.metrics['test'].items()},
            norm_drop=norm_drop,
            relative_norm_drop=relative_norm_drop,
            pruning_instability=pruning_instability,
            pruning_stability=pruning_stability,
            train_loss_increase=train_loss_increase,
            relative_train_loss_increase_factor=relative_train_loss_increase_factor,
        )

    def restore_model(self) -> None:
        """Restores the model from the checkpoint file."""
        sys.stdout.write(
            f"Restoring model from {self.checkpoint_file}.\n")
        self.model = self.get_model(reinit=False, temporary=True)

    def save_model(self, model_type: str, remove_pruning_hooks: bool = False, temporary: bool = False) -> str:
        """Saves the model to a file."""
        if model_type not in ['initial', 'trained']:
            print(f"Ignoring to save {model_type} for now.")
            return None
        fName = f"{model_type}_model.pt"
        fPath = os.path.join(wandb.run.dir, fName) if not temporary else os.path.join(self.tmp_dir, fName)
        if remove_pruning_hooks:
            self.strategy.make_pruning_permanent(model=self.model)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict
        return fPath

    def evaluate_model(self, data='train'):
        """Evaluate the model on the given data set."""
        return self.train_epoch(data=data, is_training=False)

    def define_retrain_schedule(self, n_epochs_finetune, pruning_sparsity):
        """Define the retraining schedule.
            - Tuneable schedules all require both an initial value as well as a warmup length
            - Fixed schedules require no additional parameters and are mere conversions such as LRW
        """
        tuneable_schedules = ['constant',  # Constant learning rate
                              'stepped',  # Stepped Budget Aware Conversion (BAC)
                              'cosine',  # Cosine from initial value -> 0
                              'linear',  # Linear from initial value -> 0
                              ]
        fixed_schedules = ['FT',  # Use last lr of original training as schedule (Han et al.), no warmup
                           'LRW',  # Learning Rate Rewinding (Renda et al.), no warmup
                           'SLR',  # Scaled Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'CLR',  # Cyclic Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'LLR',  # Linear from the largest original lr to 0, maxLR init, 10% warmup
                           'ALLR',  # LLR, but in the last phase behave like LCN
                           ]

        # Define the initial lr, max lr and min lr
        maxLR = max(
            self.strategy.lr_dict.values())
        after_warmup_index = (self.config.n_epochs_warmup or 0) * len(self.trainLoader)
        minLR = min(list(self.strategy.lr_dict.values())[after_warmup_index:])  # Ignores warmup in orig. schedule

        n_total_iterations = len(self.trainLoader) * n_epochs_finetune
        if self.config.retrain_schedule in tuneable_schedules:
            assert self.config.retrain_schedule_init is not None
            assert self.config.retrain_schedule_warmup is not None

            n_warmup_iterations = int(self.config.retrain_schedule_warmup * n_total_iterations)
            after_warmup_lr = self.config.retrain_schedule_init
        elif self.config.retrain_schedule in fixed_schedules:
            assert self.config.retrain_schedule_init is None
            assert self.config.retrain_schedule_warmup is None

            # Define warmup length
            if self.config.retrain_schedule in ['FT', 'LRW']:
                n_warmup_iterations = 0
            else:
                # 10% warmup
                n_warmup_iterations = int(0.1 * n_total_iterations)

            # Define the after_warmup_lr
            if self.config.retrain_schedule == 'FT':
                after_warmup_lr = minLR
            elif self.config.retrain_schedule == 'LRW':
                after_warmup_lr = list(self.strategy.lr_dict.values())[
                    -n_total_iterations]  # == remaining iterations since we don't do warmup
            elif self.config.retrain_schedule == 'ALLR':
                minLRThreshold = min(float(n_epochs_finetune) / self.config.n_epochs, 1.0) * maxLR
                # Use the norm drop
                relative_norm_drop = self.after_pruning_metrics['relative_norm_drop']
                scaling = relative_norm_drop / sqrt(pruning_sparsity)
                discounted_LR = float(scaling) * maxLR

                after_warmup_lr = np.clip(discounted_LR, a_min=minLRThreshold, a_max=maxLR)

            elif self.config.retrain_schedule in ['SLR', 'CLR', 'LLR']:
                after_warmup_lr = maxLR
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Set the optimizer lr
        for param_group in self.optimizer.param_groups:
            if n_warmup_iterations > 0:
                # If warmup, then we actually begin with 0 and increase to after_warmup_lr
                param_group['lr'] = 0.0
            else:
                param_group['lr'] = after_warmup_lr

        # Define warmup scheduler
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_warmup_iterations, eta_min=after_warmup_lr)
            milestone = n_warmup_iterations + 1

        # Define scheduler after the warmup
        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = None
        if self.config.retrain_schedule in ['FT', 'constant']:
            # Does essentially nothing but keeping the smallest learning rate
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif self.config.retrain_schedule == 'LRW':
            iterationsLR = list(self.strategy.lr_dict.values())[(-n_remaining_iterations):]
            iterationsLR.append(iterationsLR[-1])  # Double the last learning rate so we avoid the IndexError
            scheduler = FixedLR(optimizer=self.optimizer, lrList=iterationsLR)

        elif self.config.retrain_schedule in ['stepped', 'SLR']:
            iterationsLR = [lr if int(it) >= after_warmup_index else maxLR
                            for it, lr in self.strategy.lr_dict.items()]

            interpolation_width = (len(self.strategy.lr_dict)) / n_remaining_iterations  # In general not an integer
            reducedLRs = [iterationsLR[int(j * interpolation_width)] for j in range(n_remaining_iterations)]
            # Add a last LR to avoid IndexError
            reducedLRs = reducedLRs + [reducedLRs[-1]]

            lr_lambda = lambda it: reducedLRs[it] / float(maxLR)  # Function returning the correct learning rate factor
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif self.config.retrain_schedule in ['CLR', 'cosine']:
            stopLR = 0. if self.config.retrain_schedule == 'cosine' else minLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_remaining_iterations, eta_min=stopLR)

        elif self.config.retrain_schedule in ['LLR', 'ALLR', 'linear', 'LossALLR', 'AccALLR']:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=0.,
                                                          total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [after_warmup_lr for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def fine_tuning(self, pruning_sparsity, n_epochs_finetune, phase=1):
        """Fine-tuning phase of the pruning strategy."""
        if n_epochs_finetune == 0:
            self.strategy.enforce_prunedness()
            # Reset squared model norm for following pruning steps
            self.squared_model_norm = Utils.get_model_norm_square(model=self.model)
            return
        n_phases = self.config.n_phases or 1

        # Reset the GradScaler for AutoCast
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))

        # Update the retrain schedule individually for every phase/cycle
        self.define_retrain_schedule(n_epochs_finetune=n_epochs_finetune,
                                     pruning_sparsity=pruning_sparsity)

        self.strategy.set_to_finetuning_phase()
        for epoch in range(1, n_epochs_finetune + 1, 1):
            self.reset_averaged_metrics()
            sys.stdout.write(
                f"\nFinetuning: phase {phase}/{n_phases} | epoch {epoch}/{n_epochs_finetune}\n")
            # Train
            t = time.time()
            self.train_epoch(data='train')
            self.evaluate_model(data='val')

            self.strategy.at_epoch_end(epoch=epoch)
            self.log(runTime=time.time() - t, finetuning=True,
                     final_logging=(epoch == n_epochs_finetune and phase == n_phases))

        # Reset squared model norm for following pruning steps
        self.strategy.enforce_prunedness()
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

    def train_epoch(self, data='train', is_training=True):
        """Train or evaluate the model for one epoch."""
        assert not (data in ['test', 'val'] and is_training), "Can't train on test/val set."
        loaderDict = {'train': self.trainLoader,
                      'val': self.valLoader,
                      'test': self.testLoader}
        loader = loaderDict[data]

        sys.stdout.write(f"Training:\n") if is_training else sys.stdout.write(
            f"Evaluation of {data} data:\n")

        with torch.set_grad_enabled(is_training):
            for x_input, y_target in Bar(loader):
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)
                y_target = y_target.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()  # Zero the gradient buffers

                itStartTime = time.time()

                self.strategy.start_forward_mode(enable_grad=is_training)
                if is_training:
                    with autocast(enabled=(self.config.use_amp is True)):

                        output = self.model.train()(x_input)
                        loss = self.loss_criterion(output, y_target)

                        loss = self.strategy.before_backward(loss=loss, weight_decay=self.config.weight_decay)

                    self.ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                    # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                    # we might add something to the grads with during_training(), this has to be split
                    self.ampGradScaler.unscale_(self.optimizer)
                    # Potentially update the gradients
                    self.strategy.during_training(trainIteration=self.trainIterationCtr)
                    self.ampGradScaler.step(self.optimizer)
                    self.ampGradScaler.update()

                    self.strategy.end_forward_mode()  # Has no effect for DPF
                    self.strategy.after_training_iteration(it=self.trainIterationCtr,
                                                           lr=float(self.optimizer.param_groups[0]['lr']))
                    self.scheduler.step()
                    self.trainIterationCtr += 1
                else:
                    with autocast(enabled=(self.config.use_amp is True)):
                        # We use train(mode=True) for the training dataset such that we do not get the drop in loss because of running average of BN
                        # Note however that this will change the running stats and consequently also slightly the evaluation of val/eval datasets
                        output = self.model.train(mode=(data == 'train'))(x_input)
                        loss = self.loss_criterion(output, y_target)

                    self.strategy.end_forward_mode()  # Has no effect for DPF
                itEndTime = time.time()
                n_img_in_iteration = len(y_target)
                ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second

                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                self.metrics[data]['accuracy'](output, y_target)
                self.metrics[data]['ips_throughput'](ips)

    def train(self):
        """Train the model."""
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))
        for epoch in range(self.config.n_epochs + 1):
            self.reset_averaged_metrics()
            sys.stdout.write(f"\n\nEpoch {epoch}/{self.config.n_epochs}\n")
            t = time.time()
            if epoch > 0:
                # Train
                self.train_epoch(data='train')
            self.evaluate_model(data='val')

            if epoch == self.config.n_epochs:
                # Do one complete evaluation on the test data set
                self.evaluate_model(data='test')

            self.strategy.at_epoch_end(epoch=epoch)

            self.log(runTime=time.time() - t)

        self.trained_test_accuracy = self.metrics['test']['accuracy'].compute()
        self.trained_train_loss = self.metrics['train']['loss'].compute()
