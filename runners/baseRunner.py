# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining
# File:         baseRunner.py
# Description:  Base Runner class, all other runners inherit from this one
# ===========================================================================
import importlib
import json
import os
import sys
import time
from collections import OrderedDict
from math import sqrt

import numpy as np
import torch
import torch.optim as optimizers
import wandb
from barbar import Bar

from config import datasetDict, trainTransformDict, testTransformDict
from metrics import metrics
from strategies import strategies
from utilities import AverageMeter, SequentialSchedulers, FixedLR, CyclicLRAdaptiveBase
from utilities import Utilities as Utils


class baseRunner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config, debug_mode=False):

        self.config = config
        self.dataParallel = True if (self.config.dataset == 'imagenet' and torch.cuda.device_count() > 1) else False
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)

        # Dynamic retraining: Catch if the retraining length is dynamic -> it is determined as follows:
        # nepochs = 160, dynamic_retrain_length=200 => Train for 160 epochs, then retrain for 200-160=40 epochs
        if self.config.dynamic_retrain_length is not None:
            assert self.config.n_phases in [None, 1], "Dynamic retraining length only work with a single phase"
            assert self.config.n_epochs_per_phase is None
            assert self.config.dynamic_retrain_length > self.config.nepochs
            assert self.config.strategy == 'Restricted_IMP'
            self.config.update({'n_epochs_per_phase': self.config.dynamic_retrain_length - self.config.nepochs},
                               allow_val_change=True)

        # Set a couple useful variables
        self.k_accuracy = 5 if self.config.dataset in ['cifar100', 'imagenet'] else 3
        self.checkpoint_file = None
        self.trained_test_accuracy = None
        self.after_pruning_metrics = {}
        self.totalTrainTime = None
        self.totalFinetuneTime = None
        self.debug_mode = debug_mode  # If active, use less desired_sparsities, etc
        self.seed = None
        self.trainedModelFile = None
        self.squared_model_norm = None
        self.n_warmup_epochs = None
        self.trainIterationCtr = 1

        # Budgeted training variables
        self.stability_scaling = None  # Factor to multiply cycle amplitude for BIMP_LC

        # Variables to be set by inheriting classes
        self.strategy = None
        self.trainLoader = None
        self.testLoader = None
        self.model = None

        # Define the loss object and metrics
        # Important note: for the correct computation of loss/accuracy it's important to have reduction == 'mean'
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

        self.train_loss, self.train_accuracy, self.train_k_accuracy = AverageMeter(), AverageMeter(), AverageMeter()
        self.test_loss, self.test_accuracy, self.test_k_accuracy = AverageMeter(), AverageMeter(), AverageMeter()

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.train_k_accuracy.reset()

        self.test_loss.reset()
        self.test_accuracy.reset()
        self.test_k_accuracy.reset()

    def get_metrics(self):
        with torch.no_grad():
            self.strategy.start_forward_mode()  # Necessary to have correct computations for DPF
            x_input, y_target = next(iter(self.testLoader))
            x_input, y_target = x_input.to(self.device), y_target.to(self.device)  # Move to CUDA if possible
            n_flops, n_nonzero_flops = metrics.get_flops(model=self.model, x_input=x_input)
            n_total, n_nonzero = metrics.get_parameter_count(model=self.model)

            loggingDict = dict(
                train=dict(
                    loss=self.train_loss.result(),
                    accuracy=self.train_accuracy.result(),
                    k_accuracy=self.train_k_accuracy.result(),
                ),
                test=dict(
                    loss=self.test_loss.result(),
                    accuracy=self.test_accuracy.result(),
                    k_accuracy=self.test_k_accuracy.result(),
                ),
                global_sparsity=metrics.global_sparsity(module=self.model),
                global_compression=metrics.compression_rate(module=self.model),
                nonzero_inference_flops=n_nonzero_flops,
                baseline_inference_flops=n_flops,
                theoretical_speedup=metrics.get_theoretical_speedup(n_flops=n_flops, n_nonzero_flops=n_nonzero_flops),
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                distance_to_origin=metrics.get_distance_to_origin(self.model),
            )
            self.strategy.end_forward_mode()  # Necessary to have correct computations for DPF
        return loggingDict

    def get_dataloaders(self):
        rootPath = f"./datasets_pytorch/{self.config.dataset}-data"

        if self.config.dataset in ['imagenet']:
            trainData = datasetDict[self.config.dataset](root=rootPath, split='train',
                                                         transform=trainTransformDict[self.config.dataset])
            testData = datasetDict[self.config.dataset](root=rootPath, split='val',
                                                        transform=testTransformDict[self.config.dataset])
        else:
            trainData = datasetDict[self.config.dataset](root=rootPath, train=True, download=True,
                                                         transform=trainTransformDict[self.config.dataset])
            testData = datasetDict[self.config.dataset](root=rootPath, train=False,
                                                        transform=testTransformDict[self.config.dataset])

        if self.config.dataset in ['imagenet', 'cifar100']:
            num_workers = 4 if torch.cuda.is_available() else 0
        else:
            num_workers = 2 if torch.cuda.is_available() else 0

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, shuffle=True,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers)
        testLoader = torch.utils.data.DataLoader(testData, batch_size=self.config.batch_size, shuffle=False,
                                                 pin_memory=torch.cuda.is_available(), num_workers=num_workers)

        return trainLoader, testLoader

    def get_model(self, load_checkpoint: bool = False, load_initial: bool = False) -> torch.nn.Module:
        if not load_checkpoint:
            # Define the model
            model = getattr(importlib.import_module('models.' + self.config.dataset), self.config.model)()
        if load_checkpoint:
            # self.model must exist already
            model = self.model
            # Note, we have to get rid of all existing prunings, otherwise we cannot load the state_dict as it is unpruned
            self.strategy.remove_pruning_hooks(model=self.model)
            file = self.checkpoint_file if not load_initial else os.path.join(wandb.run.dir, self.trainedModelFile)

            # We need to check whether the model was loaded using dataparallel, in that case remove 'module'
            # original saved file with DataParallel
            state_dict = torch.load(
                file, map_location=self.device)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                new_key = key
                if key.startswith("module."):
                    new_key = key[7:]  # remove `module.`
                new_state_dict[new_key] = val
            model.load_state_dict(new_state_dict)
        elif load_initial:
            # Load initial model from specified path

            # We need to check whether the model was loaded using dataparallel, in that case remove 'module'
            # original saved file with DataParallel
            state_dict = torch.load(os.path.join(wandb.run.dir, self.trainedModelFile), map_location=self.device)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                new_key = key
                if key.startswith("module."):
                    new_key = key[7:]  # remove `module.`
                new_state_dict[new_key] = val
            # load params
            model.load_state_dict(new_state_dict)

        if self.dataParallel and not load_checkpoint:  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def define_optimizer_scheduler(self):
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = self.config.learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant', 'Cyclic', 'CyclicDecay',
                    'CyclicBudget', 'CyclicBudgetDecay', 'CyclicBudgetDecayBase', 'CyclicBudgetLC']:
            scheduler = (name, kwargs)
            self.initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        wd = self.config.weight_decay if self.config.strategy != 'CS' else 0.  # For CS, apply wd manually
        self.optimizer = optimizers.SGD(params=self.model.parameters(), lr=self.initial_lr,
                                        momentum=self.config.momentum,
                                        weight_decay=wd, nesterov=self.config.nesterov)

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = len(self.trainLoader)
        n_total_iterations = iterations_per_epoch * self.config.nepochs
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
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=0.,
                                                          total_iters=n_remaining_iterations)
        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        elif name in ['Cyclic', 'CyclicDecay']:
            assert self.config.strategy == 'GMP', "Cyclic LR works only with GMP for now."
            assert self.config.n_epochs_warmup in [None, 0], "Cyclic LR does not work with network warmup for now."

            cycle_warmup_length = int(0.1 * iterations_per_epoch * self.config.pruning_interval)
            cycle_length = int(iterations_per_epoch * self.config.pruning_interval) - cycle_warmup_length

            linear_decay_scale_fn, last_bit_decay_fn = None, None
            if name == 'CyclicDecay':
                # We linearly shrinken the triangles
                n_cycles = self.strategy.n_pruning_steps + 1

                def linear_decay_scale_fn(x):
                    # x == cycle iteration count
                    y = 1 - float(x - 1) / n_cycles
                    return y

                def last_bit_decay_fn(x):
                    # Function for last part CyclicLR
                    y = 1 - float(n_cycles - 1) / n_cycles
                    return y

            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                                          step_size_up=cycle_warmup_length, step_size_down=cycle_length,
                                                          cycle_momentum=False, base_momentum=self.config.momentum,
                                                          scale_fn=linear_decay_scale_fn,
                                                          max_momentum=self.config.momentum)
            change_milestone = int(
                iterations_per_epoch * self.strategy.pruning_interval * self.strategy.n_pruning_steps) + 1

            # Define the scheduler for the last bit
            remaining_epochs = self.config.nepochs - self.strategy.pruning_interval * self.strategy.n_pruning_steps
            cycle_length = int(iterations_per_epoch * remaining_epochs) - cycle_warmup_length
            last_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                                               step_size_up=cycle_warmup_length,
                                                               step_size_down=cycle_length,
                                                               cycle_momentum=False, base_momentum=self.config.momentum,
                                                               scale_fn=last_bit_decay_fn,
                                                               max_momentum=self.config.momentum)

            scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[scheduler, last_scheduler],
                                             milestones=[change_milestone])

        elif name in ['CyclicBudget', 'CyclicBudgetDecay', 'CyclicBudgetDecayBase']:
            assert self.config.strategy == 'BIMP', "CyclicBudget LR only works with BIMP_LC for now."
            assert self.config.n_epochs_warmup in [None,
                                                   0], "CyclicBudget LR does not work with network warmup for now."

            # Define Linear scheduler for budgeted training
            budgeted_training_iterations = int(iterations_per_epoch * self.config.n_train_budget)
            budget_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1.0, end_factor=0.,
                                                                 total_iters=budgeted_training_iterations)
            budget_change_milestone = budgeted_training_iterations + 1

            linear_decay_scale_fn, last_bit_decay_fn = None, None
            base_lr_scale_fn = None
            n_cycles = self.strategy.n_pruning_steps
            if name == 'CyclicBudgetDecay':
                # We linearly shrinken the triangles
                def linear_decay_scale_fn(x):
                    # x == cycle iteration count
                    y = 1 - float(x - 1) / n_cycles
                    return y

                def last_bit_decay_fn(x):
                    # Function for last part CyclicLR
                    y = 1 - float(n_cycles - 1) / n_cycles
                    return y
            elif name == 'CyclicBudgetDecayBase':
                # We linearly shrink the base lr
                def base_lr_scale_fn(x):
                    # x == cycle iteration count
                    y = 1 - float(x - 1) / (n_cycles - 1)
                    return y

            # Define the retraining-cycle scheduler
            cycle_warmup_length = int(0.1 * iterations_per_epoch * self.config.pruning_interval)
            cycle_length = int(iterations_per_epoch * self.config.pruning_interval) - cycle_warmup_length

            scheduler = CyclicLRAdaptiveBase(optimizer=self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                             step_size_up=cycle_warmup_length, step_size_down=cycle_length,
                                             cycle_momentum=False, base_momentum=self.config.momentum,
                                             scale_fn=linear_decay_scale_fn,
                                             base_lr_scale_fn=base_lr_scale_fn,
                                             max_momentum=self.config.momentum)
            last_change_milestone = budgeted_training_iterations + int(
                iterations_per_epoch * self.strategy.pruning_interval * (self.strategy.n_pruning_steps - 1)) + 1

            # Define the Scheduler for the last bit
            remaining_epochs = self.config.nepochs - self.config.n_train_budget - self.strategy.pruning_interval * (
                    self.strategy.n_pruning_steps - 1)
            cycle_length = int(iterations_per_epoch * remaining_epochs) - cycle_warmup_length
            last_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                                               step_size_up=cycle_warmup_length,
                                                               step_size_down=cycle_length,
                                                               cycle_momentum=False, base_momentum=self.config.momentum,
                                                               scale_fn=last_bit_decay_fn,
                                                               max_momentum=self.config.momentum)

            scheduler = SequentialSchedulers(optimizer=self.optimizer,
                                             schedulers=[budget_scheduler, scheduler, last_scheduler],
                                             milestones=[budget_change_milestone, last_change_milestone])
        elif name == 'CyclicBudgetLC':
            assert self.config.strategy in ['BIMP_LC', 'BIMP_LCT', 'BIMP_LCN', 'BIMP_LCNT',
                                            'BIMP_ALLR'], "CyclicBudgetLC LR only works with BIMP for now."
            assert self.config.n_epochs_warmup in [None,
                                                   0], "CyclicBudgetLC LR does not work with network warmup for now."
            assert self.config.pruning_interval <= self.config.n_train_budget, "Pruning interval too large, can't employ the LRW logic of Copycat."

            # Define Linear scheduler for budgeted training
            budgeted_training_iterations = int(iterations_per_epoch * self.config.n_train_budget)
            budget_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1.0, end_factor=0.,
                                                                 total_iters=budgeted_training_iterations)
            budget_change_milestone = budgeted_training_iterations + 1

            # We determine the cycle amplitude by using the closest train loss from the budgeted training phase
            # n_cycles = self.strategy.n_pruning_steps + 1
            def LC_scale_fn(x):
                # x == cycle iteration count
                return self.stability_scaling or 1.0

            # Define the retraining-cycle scheduler
            cycle_warmup_length = int(0.1 * iterations_per_epoch * self.config.pruning_interval)
            cycle_length = int(iterations_per_epoch * self.config.pruning_interval) - cycle_warmup_length

            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                                          step_size_up=cycle_warmup_length, step_size_down=cycle_length,
                                                          cycle_momentum=False, base_momentum=self.config.momentum,
                                                          scale_fn=LC_scale_fn,
                                                          max_momentum=self.config.momentum)
            last_change_milestone = budgeted_training_iterations + int(
                iterations_per_epoch * self.strategy.pruning_interval * (self.strategy.n_pruning_steps - 1)) + 1

            # Define the Scheduler for the last bit
            remaining_epochs = self.config.nepochs - self.config.n_train_budget - self.strategy.pruning_interval * (
                    self.strategy.n_pruning_steps - 1)
            cycle_length = int(iterations_per_epoch * remaining_epochs) - cycle_warmup_length
            last_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0., max_lr=self.initial_lr,
                                                               step_size_up=cycle_warmup_length,
                                                               step_size_down=cycle_length,
                                                               cycle_momentum=False, base_momentum=self.config.momentum,
                                                               scale_fn=LC_scale_fn,
                                                               max_momentum=self.config.momentum)

            scheduler = SequentialSchedulers(optimizer=self.optimizer,
                                             schedulers=[budget_scheduler, scheduler, last_scheduler],
                                             milestones=[budget_change_milestone, last_change_milestone])
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
        assert self.config.n_phases in [None, 1] or self.config.IMP_selector
        if self.config.strategy == 'Dense':
            return strategies.Dense()

        elif self.config.strategy in ['IMP', 'Restricted_IMP']:
            assert self.config.IMP_selector in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
            # Select the correct class using the IMP_selector variable
            selector_to_class = {
                'global': strategies.IMP,
                'uniform': strategies.IMP_Uniform,
                'uniform_plus': strategies.IMP_UniformPlus,
                'ERK': strategies.IMP_ERK,
                'LAMP': strategies.IMP_LAMP,
            }
            return selector_to_class[self.config.IMP_selector](desired_sparsity=self.config.goal_sparsity,
                                                               n_phases=self.config.n_phases,
                                                               n_epochs_per_phase=self.config.n_epochs_per_phase)
        elif self.config.strategy == 'GMP':
            assert self.config.GMP_selector in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']

            # Select the correct class using the GMP_selector variable
            selector_to_class = {
                'global': strategies.GMP,
                'uniform': strategies.GMP_Uniform,
                'uniform_plus': strategies.GMP_UniformPlus,
                'ERK': strategies.GMP_ERK,
                'LAMP': strategies.GMP_LAMP
            }
            return selector_to_class[self.config.GMP_selector](model=self.model, n_train_epochs=self.config.nepochs,
                                                               n_epochs_per_phase=self.config.n_epochs_per_phase,
                                                               desired_sparsity=self.config.goal_sparsity,
                                                               pruning_interval=self.config.pruning_interval,
                                                               allow_recovering=self.config.allow_recovering,
                                                               after_pruning_callback=self.after_pruning_callback)
        elif self.config.strategy == 'BIMP':
            return strategies.BIMP(model=self.model, n_train_epochs=self.config.nepochs,
                                   n_train_budget=self.config.n_train_budget,
                                   n_epochs_per_phase=self.config.n_epochs_per_phase,
                                   desired_sparsity=self.config.goal_sparsity,
                                   pruning_interval=self.config.pruning_interval,
                                   after_pruning_callback=self.after_pruning_callback)
        elif self.config.strategy in ['BIMP_LC', 'BIMP_LCT']:
            return strategies.BIMP_LC(model=self.model, n_train_epochs=self.config.nepochs,
                                      n_train_budget=self.config.n_train_budget,
                                      n_epochs_per_phase=self.config.n_epochs_per_phase,
                                      desired_sparsity=self.config.goal_sparsity,
                                      pruning_interval=self.config.pruning_interval,
                                      after_pruning_callback=self.after_pruning_callback)
        elif self.config.strategy in ['BIMP_LCN', 'BIMP_LCNT', 'BIMP_ALLR']:
            return strategies.BIMP_LCN(scaling_callback=self.set_stability_scaling_LCN,
                                       model=self.model, n_train_epochs=self.config.nepochs,
                                       n_train_budget=self.config.n_train_budget,
                                       n_epochs_per_phase=self.config.n_epochs_per_phase,
                                       desired_sparsity=self.config.goal_sparsity,
                                       pruning_interval=self.config.pruning_interval,
                                       after_pruning_callback=self.after_pruning_callback)

        elif self.config.strategy == 'GSM':
            return strategies.GSM(desired_sparsity=self.config.goal_sparsity,
                                  n_epochs_per_phase=self.config.n_epochs_per_phase)
        elif self.config.strategy == 'LC':
            return strategies.LC(desired_sparsity=self.config.goal_sparsity,
                                 n_epochs_per_phase=self.config.n_epochs_per_phase,
                                 change_weight_decay_callback=self.change_weight_decay_callback,
                                 n_epochs_total=self.config.nepochs,
                                 initial_weight_decay=self.config.weight_decay)

        elif self.config.strategy == 'DPF':
            return strategies.DPF(model=self.model, n_train_epochs=self.config.nepochs,
                                  n_epochs_per_phase=self.config.n_epochs_per_phase,
                                  desired_sparsity=self.config.goal_sparsity,
                                  pruning_interval=self.config.pruning_interval,
                                  after_pruning_callback=self.after_pruning_callback)

        elif self.config.strategy == 'CS':
            return strategies.CS(n_epochs_per_phase=self.config.n_epochs_per_phase, s_initial=self.config.s_initial,
                                 beta_final=self.config.beta_final,
                                 T_it=int(len(self.trainLoader) * self.config.nepochs))
        elif self.config.strategy == 'STR':
            return strategies.STR(n_epochs_per_phase=self.config.n_epochs_per_phase, s_initial=self.config.s_initial,
                                  use_global_threshold=self.config.use_global_threshold)
        elif self.config.strategy == 'DST':
            return strategies.DST(n_epochs_per_phase=self.config.n_epochs_per_phase)
        elif self.config.strategy == 'DNW':
            return strategies.DNW(desired_sparsity=self.config.goal_sparsity,
                                  n_epochs_per_phase=self.config.n_epochs_per_phase)

    def log(self, runTime, finetuning: bool = False, desired_sparsity=None):
        loggingDict = self.get_metrics()
        self.strategy.start_forward_mode()
        loggingDict.update({'epoch_run_time': runTime})
        if not finetuning:
            if self.config.goal_sparsity is not None:
                distance_to_pruned, rel_distance_to_pruned = metrics.get_distance_to_pruned(model=self.model,
                                                                                            sparsity=self.config.goal_sparsity)
                loggingDict.update({'distance_to_pruned': distance_to_pruned,
                                    'relative_distance_to_pruned': rel_distance_to_pruned})

            # Update final trained metrics (necessary to be able to filter via wandb)
            for metric_type, val in loggingDict.items():
                wandb.run.summary[f"trained.{metric_type}"] = val
            if self.totalTrainTime:
                # Total train time captured, hence training is done
                wandb.run.summary["trained.total_train_time"] = self.totalTrainTime
            # The usual logging of one epoch
            wandb.log(
                loggingDict
            )

        else:
            if desired_sparsity is not None:
                finalDict = dict(finetune=loggingDict,
                                 pruned=self.after_pruning_metrics[desired_sparsity],  # Metrics directly after pruning
                                 desired_sparsity=desired_sparsity,
                                 total_finetune_time=self.totalFinetuneTime,
                                 )
                wandb.log(finalDict)
                # Dump sparsity distribution to json and upload
                sparsity_distribution = metrics.per_layer_sparsity(model=self.model)
                fPath = os.path.join(wandb.run.dir, f'sparsity_distribution_{desired_sparsity}.json')
                with open(fPath, 'w') as fp:
                    json.dump(sparsity_distribution, fp)
                wandb.save(fPath)

            else:
                wandb.log(
                    dict(finetune=loggingDict,
                         ),
                )
        self.strategy.end_forward_mode()

    def final_log(self, actual_sparsity=None):
        s = self.config.goal_sparsity
        if actual_sparsity is not None and self.config.goal_sparsity is None:
            wandb.run.summary[f"actual_sparsity"] = actual_sparsity
            s = actual_sparsity
        elif self.config.goal_sparsity is None:
            # Note: This function may only be called if a desired_sparsity has been given upfront, i.e. GSM etc
            raise AssertionError("Final logging was called even though no goal_sparsity was given.")

        # Recompute accuracy and loss
        sys.stdout.write(
            f"\nFinal logging\n")
        self.reset_averaged_metrics()
        self.evaluate_model(data='train')
        self.evaluate_model(data='test')
        # Update final trained metrics (necessary to be able to filter via wandb)
        loggingDict = self.get_metrics()
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"final.{metric_type}"] = val

        # Update after prune metrics
        if s in self.after_pruning_metrics:
            for metric_type, val in self.after_pruning_metrics[s].items():
                wandb.run.summary[f"pruned.{metric_type}"] = val

    def set_stability_scaling_LCN(self, before_pruning, pruning_sparsity, is_last_pruning):
        # Sets self.stability_scaling appropriately, only works with LCN
        if before_pruning:
            # Just compute the norm
            self.squared_model_norm = Utils.get_model_norm_square(model=self.model)
        else:
            L2_norm_square = Utils.get_model_norm_square(self.model)
            norm_drop = sqrt(abs(self.squared_model_norm - L2_norm_square))
            relative_norm_drop = norm_drop / float(sqrt(self.squared_model_norm))
            norm_scaling = relative_norm_drop / sqrt(pruning_sparsity)
            if self.config.strategy == 'BIMP_LCN':
                remaining_epochs = self.config.nepochs - list(self.strategy.train_loss_dict.keys())[-1]
                minimumScaling = min(float(remaining_epochs) / self.config.n_train_budget, 1.0)
            elif self.config.strategy == 'BIMP_LCNT':
                minimumScaling = 1. / 100
            elif self.config.strategy == 'BIMP_ALLR':
                if is_last_pruning:
                    remaining_epochs = self.config.nepochs - list(self.strategy.train_loss_dict.keys())[-1]
                    minimumScaling = min(float(remaining_epochs) / self.config.n_train_budget, 1.0)
                else:
                    minimumScaling = 1.0

            self.stability_scaling = np.clip(norm_scaling, a_min=minimumScaling, a_max=1.0)
            print(
                f"Minimum Scaling {minimumScaling} | Relative Norm Drop {relative_norm_drop} | After correction {norm_scaling} | Final Stability Scaling {self.stability_scaling}")

    def after_pruning_callback(self, desired_sparsity: float, prune_momentum: bool = False,
                               reset_momentum: bool = False) -> None:
        # This function must be called once for every sparsity, directly after pruning
        # Compute losses, accuracies after pruning
        sys.stdout.write(f"\nDesired sparsity {desired_sparsity} - Computing incurred losses after pruning.\n")
        before_pruning_train_accuracy = self.train_accuracy.result()
        self.reset_averaged_metrics()
        self.evaluate_model(data='train')
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

        # Set budgeted train stability
        if self.config.strategy in ['BIMP_LC', 'BIMP_LCT']:
            budget_train_accuracy = before_pruning_train_accuracy
            current_train_accuracy = self.train_accuracy.result()
            budgeted_train_instability = (budget_train_accuracy - current_train_accuracy) / budget_train_accuracy
            if self.config.strategy == 'BIMP_LC':
                # Get the number of remaining retrain epochs (over all cycles!)
                remaining_epochs = self.config.nepochs - list(self.strategy.train_loss_dict.keys())[-1]
                minimumScaling = min(float(remaining_epochs) / self.config.n_train_budget, 1.0)
            elif self.config.strategy == 'BIMP_LCT':
                minimumScaling = 1. / 100

            self.stability_scaling = np.clip(budgeted_train_instability, a_min=minimumScaling, a_max=1.0)
            print(budgeted_train_instability, minimumScaling, self.stability_scaling)

        if self.trained_test_accuracy is not None and self.trained_test_accuracy > 0:
            pruning_instability = (
                                          self.trained_test_accuracy - self.test_accuracy.result()) / self.trained_test_accuracy
            pruning_stability = 1 - pruning_instability
        else:
            pruning_instability, pruning_stability = {}, {}
        self.after_pruning_metrics[desired_sparsity] = dict(
            train=dict(
                loss=self.train_loss.result(),
                accuracy=self.train_accuracy.result(),
                k_accuracy=self.train_k_accuracy.result(),
            ),
            test=dict(
                loss=self.test_loss.result(),
                accuracy=self.test_accuracy.result(),
                k_accuracy=self.test_k_accuracy.result(),
            ),
            norm_drop=norm_drop,
            relative_norm_drop=relative_norm_drop,
            pruning_instability=pruning_instability,
            pruning_stability=pruning_stability,
        )
        if reset_momentum:
            sys.stdout.write(
                f"Resetting momentum_buffer (if existing) for potential finetuning.\n")
            self.strategy.reset_momentum(optimizer=self.optimizer)
        elif prune_momentum:
            sys.stdout.write(
                f"Pruning momentum_buffer (if existing).\n")
            self.strategy.prune_momentum(optimizer=self.optimizer)

    def change_weight_decay_callback(self, penalty):
        for group in self.optimizer.param_groups:
            group['weight_decay'] = penalty
        print(f"Changed weight decay to {penalty}.")

    def restore_model(self, from_initial=False) -> None:
        outputStr = 'initial' if from_initial else 'checkpoint'
        sys.stdout.write(
            f"Restoring {outputStr} model from {self.checkpoint_file if not from_initial else self.trainedModelFile}.\n")
        self.model = self.get_model(load_checkpoint=True, load_initial=(from_initial is True))

    def save_model(self, model_type: str, remove_pruning_hooks: bool = False) -> str:
        if model_type == 'initial':
            fName = self.trainedModelFile
        else:
            fName = f"{self.config.dataset}_SGD_{self.config.model}_{model_type}_{self.config.run_id}_{self.seed}.pt"
        if model_type == 'trained':
            # Save the trained model name to wandb
            wandb.summary['trained_model_file'] = fName
        fPath = os.path.join(wandb.run.dir, fName)
        if remove_pruning_hooks:
            self.strategy.remove_pruning_hooks(model=self.model)
        torch.save(self.model.state_dict(), fPath)  # Save the state_dict to the wandb directory
        return fPath

    def evaluate_model(self, data='train'):
        return self.train_epoch(data=data, is_training=False)

    def define_retrain_schedule(self, n_epochs_finetune, desired_sparsity, pruning_sparsity, phase):
        """Define the retraining schedule.
            - Tuneable schedules all require both an initial value as well as a warmup length
            - Fixed schedules require no additional parameters and are mere conversions such as LRW
        """
        tuneable_schedules = ['constant',  # Constant learning rate
                              'stepped',  # Stepped Budget Aware Conversion (BAC)
                              'cosine',  # Cosine from initial value -> 0
                              'cosine_min',  # Same as 'cosine', but -> smallest original lr
                              'linear',  # Linear from initial value -> 0
                              'linear_min'  # Same as 'linear', but -> smallest original lr
                              ]
        fixed_schedules = ['FT',  # Use last lr of original training as schedule (Han et al.), no warmup
                           'LRW',  # Learning Rate Rewinding (Renda et al.), no warmup
                           'SLR',  # Scaled Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'CLR',  # Cyclic Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'LLR',  # Linear from the largest original lr to 0, maxLR init, 10% warmup
                           'LLR_min',  # Same as CLR but linear, i.e. going to minLR
                           'LC',
                           # Same as LLR, but the initial lr depends on the pruning instability, further uses a heuristic to determine the minimal learning rate from the available retraining time, 10% warmup
                           'LCT',  # Same as LC, but uses a minimum threshold on the LR, 10% warmup,
                           'LCN',
                           # Same as LC, but uses the relative norm drop instead of pruning instability, 10% warmup,
                           'ALLR',  # LLR, but in the last epoch behave like LCN
                           ]

        # Define the initial lr, max lr and min lr
        maxLR = max(self.strategy.lr_dict.values())
        after_warmup_index = self.config.n_epochs_warmup or 0
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
                after_warmup_lr = self.strategy.lr_dict[self.config.nepochs - n_epochs_finetune + 1]
            elif self.config.retrain_schedule in ['LC', 'LCT', 'LCN', 'ALLR']:
                if self.config.retrain_schedule in ['LC', 'LCN']:
                    # Selection similar to LRW, but over all remaining epochs
                    if self.config.n_phases == 1:
                        minLRThreshold = min(float(n_epochs_finetune) / self.config.nepochs, 1.0) * maxLR
                    elif self.config.n_phases > 1:
                        ep_remaining = self.config.n_epochs_per_phase * self.config.n_phases - (
                                phase - 1) * self.config.n_epochs_per_phase
                        minLRThreshold = min(float(ep_remaining) / self.config.nepochs, 1.0) * maxLR

                elif self.config.retrain_schedule == 'LCT':
                    minLRThreshold = float(maxLR) / 100

                elif self.config.retrain_schedule == 'ALLR':
                    if phase == self.config.n_phases:
                        # Last phase, so do LCN
                        minLRThreshold = min(float(n_epochs_finetune) / self.config.nepochs, 1.0) * maxLR
                    else:
                        minLRThreshold = maxLR

                # Discounting approach
                if self.config.retrain_schedule in ['LCN', 'ALLR']:
                    # Use the norm drop
                    relative_norm_drop = self.after_pruning_metrics[desired_sparsity]['relative_norm_drop']
                    norm_scaling = relative_norm_drop / sqrt(pruning_sparsity)
                    discounted_LR = norm_scaling * maxLR
                else:
                    # Use the instability
                    pruned_train_acc = self.after_pruning_metrics[desired_sparsity]['train']['accuracy']
                    reference_train_acc = self.strategy.train_acc_dict[self.config.nepochs]
                    train_instability = (reference_train_acc - pruned_train_acc) / reference_train_acc
                    discounted_LR = train_instability * maxLR
                after_warmup_lr = np.clip(discounted_LR, a_min=minLRThreshold, a_max=maxLR)

            elif self.config.retrain_schedule in ['SLR', 'CLR', 'LLR', 'LLR_min']:
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
            rewind_epoch = self.config.nepochs - n_epochs_finetune + 1
            # Convert original schedule to iterations
            iterationsLR = []
            for ep in range(rewind_epoch, self.config.nepochs + 1, 1):
                lr = self.strategy.lr_dict[ep]
                iterationsLR = iterationsLR + len(self.trainLoader) * [lr]
            iterationsLR = iterationsLR[(-n_remaining_iterations):]
            iterationsLR.append(iterationsLR[-1])  # Double the last learning rate so we avoid the IndexError
            scheduler = FixedLR(optimizer=self.optimizer, lrList=iterationsLR)

        elif self.config.retrain_schedule in ['stepped', 'SLR']:
            epochLR = [self.strategy.lr_dict[i] if i >= after_warmup_index else maxLR
                       for i in range(1, self.config.nepochs + 1, 1)]
            # Convert original schedule to iterations
            iterationsLR = []
            for lr in epochLR:
                iterationsLR = iterationsLR + len(self.trainLoader) * [lr]

            interpolation_width = (len(self.trainLoader) * len(
                epochLR)) / n_remaining_iterations  # In general not an integer
            reducedLRs = [iterationsLR[int(j * interpolation_width)] for j in range(n_remaining_iterations)]
            # Add a last LR to avoid IndexError
            reducedLRs = reducedLRs + [reducedLRs[-1]]

            lr_lambda = lambda it: reducedLRs[it] / float(maxLR)  # Function returning the correct learning rate factor
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif self.config.retrain_schedule in ['CLR', 'cosine_min', 'cosine']:
            stopLR = 0. if self.config.retrain_schedule == 'cosine' else minLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_remaining_iterations, eta_min=stopLR)

        elif self.config.retrain_schedule in ['LLR', 'LC', 'LCT', 'LCN', 'ALLR', 'linear_min', 'linear', 'LLR_min']:
            if self.config.retrain_schedule in ['linear_min', 'LLR_min']:
                endFactor = minLR / float(after_warmup_lr)
                endFactor = min(endFactor, 1.0)  # Do not let the LR become bigger
            else:
                endFactor = 0.

            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=endFactor,
                                                          total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [after_warmup_lr for group in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def fine_tuning(self, desired_sparsity, n_epochs_finetune, phase=1):
        n_phases = self.config.n_phases or 1
        # Get the pruning sparsity to use it in LCN if necessary
        if self.config.n_phases is not None:
            pruning_sparsity = 1 - (1 - self.config.goal_sparsity) ** (1. / self.config.n_phases)
        else:
            # Iterative Renda case
            pruning_sparsity = 0.2

        if phase == 1:
            self.finetuneStartTime = time.time()

        # Update the retrain schedule individually for every phase/cycle
        self.define_retrain_schedule(n_epochs_finetune=n_epochs_finetune, desired_sparsity=desired_sparsity,
                                     pruning_sparsity=pruning_sparsity, phase=phase)

        self.strategy.set_to_finetuning_phase()
        for epoch in range(1, n_epochs_finetune + 1, 1):
            self.reset_averaged_metrics()
            sys.stdout.write(
                f"\nDesired sparsity {desired_sparsity} - Finetuning: phase {phase}/{n_phases} | epoch {epoch}/{n_epochs_finetune}\n")
            # Train
            t = time.time()
            self.train_epoch(data='train')
            self.evaluate_model(data='test')
            sys.stdout.write(
                f"\nTest accuracy after this epoch: {self.test_accuracy.result()} (lr = {float(self.optimizer.param_groups[0]['lr'])})\n")

            if epoch == n_epochs_finetune and phase == n_phases:
                # Training complete, log the time
                self.totalFinetuneTime = time.time() - self.finetuneStartTime

            # As opposed to previous runs, push information every epoch, but only link to desired_sparsity at end
            dsParam = None
            if epoch == n_epochs_finetune and phase == n_phases:
                dsParam = desired_sparsity
            self.log(runTime=time.time() - t, finetuning=True, desired_sparsity=dsParam)

    def train_epoch(self, data='train', is_training=True):
        assert not (data == 'test' and is_training), "Can't train on test set."
        if data == 'train':
            loader = self.trainLoader
            mean_loss, mean_accuracy, mean_k_accuracy = self.train_loss, self.train_accuracy, self.train_k_accuracy
        elif data == 'test':
            loader = self.testLoader
            mean_loss, mean_accuracy, mean_k_accuracy = self.test_loss, self.test_accuracy, self.test_k_accuracy

        if is_training:
            sys.stdout.write(f"Training:\n")
        else:
            sys.stdout.write(f"Evaluation of {data} data:\n")

        with torch.set_grad_enabled(is_training):
            for x_input, y_target in Bar(loader):
                x_input, y_target = x_input.to(self.device), y_target.to(self.device)  # Move to CUDA if possible
                self.optimizer.zero_grad()  # Zero the gradient buffers
                self.strategy.start_forward_mode(enable_grad=is_training)
                if is_training:
                    output = self.model.train()(x_input)
                    loss = self.loss_criterion(output, y_target)
                    loss = self.strategy.before_backward(loss=loss, weight_decay=self.config.weight_decay,
                                                         penalty=self.config.lmbd)
                    loss.backward()  # Backpropagation
                    self.strategy.during_training(opt=self.optimizer, trainIteration=self.trainIterationCtr)
                    self.optimizer.step()
                    self.strategy.end_forward_mode()  # Has no effect for DPF
                    self.strategy.after_training_iteration(it=self.trainIterationCtr)
                    self.scheduler.step()
                else:
                    output = self.model.eval()(x_input)
                    loss = self.loss_criterion(output, y_target)
                    self.strategy.end_forward_mode()  # Has no effect for DPF

                mean_loss(loss.item(), len(y_target))
                mean_accuracy(Utils.categorical_accuracy(y_true=y_target, output=output), len(y_target))
                mean_k_accuracy(Utils.categorical_accuracy(y_true=y_target, output=output, topk=self.k_accuracy),
                                len(y_target))

    def train(self):
        trainStartTime = time.time()
        for epoch in range(self.config.nepochs + 1):
            self.reset_averaged_metrics()
            sys.stdout.write(f"\n\nEpoch {epoch}/{self.config.nepochs}\n")
            t = time.time()
            if epoch == 0:
                # Just evaluate the model once to get the metrics
                if self.debug_mode:
                    # Skip this step
                    sys.stdout.write(f"Skipping since we are in debug mode")
                    continue
                self.evaluate_model(data='train')
                epoch_lr = float(self.optimizer.param_groups[0]['lr'])
            else:
                # Train
                self.train_epoch(data='train')
                # Save the learning rate for potential rewinding before updating
                epoch_lr = float(self.optimizer.param_groups[0]['lr'])
            self.evaluate_model(data='test')

            self.strategy.at_epoch_end(epoch=epoch, epoch_lr=epoch_lr, train_loss=self.train_loss.result(),
                                       train_acc=self.train_accuracy.result(), optimizer=self.optimizer)

            # Save info to wandb
            if epoch == self.config.nepochs:
                # Training complete, log the time
                self.totalTrainTime = time.time() - trainStartTime
            self.log(runTime=time.time() - t)
        self.trained_test_accuracy = self.test_accuracy.result()
