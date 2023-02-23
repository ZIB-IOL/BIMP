# ===========================================================================
# Project:      How I Learned to Stop Worrying and Love Retraining - IOL Lab @ ZIB
# File:         utilities.py
# Description:  Contains some useful functions.
# ===========================================================================
import json
import os

import torch
import wandb


class Utilities:
    """Class of utility functions"""

    @staticmethod
    @torch.no_grad()
    def get_model_norm_square(model: torch.nn.Module):
        """Get L2 norm squared of parameter vector. This works for a pruned model as well."""
        squared_norm = 0.
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    param = getattr(module, param_type)
                    squared_norm += torch.norm(param, p=2) ** 2
        return float(squared_norm)

    @staticmethod
    def dump_dict_to_json_wandb(dumpDict: dict, name: str):
        """Dump some dict to json and upload it."""
        fPath = os.path.join(wandb.run.dir, f'{name}.json')
        with open(fPath, 'w') as fp:
            json.dump(dumpDict, fp)
        wandb.save(fPath)
