"""
Runs a sweep over the specified file.
To use, specify `config`, `sweep_config`, `fsdp_config`, and `script_name` arguments.
"""

import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
import time


def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list


def run(cli_args):
    if "debug" in cli_args:
        print("Debug Mode")
        slurm_job_id = 0
        slurm_task_id = 1
    else:
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    # Compute overrides
    base_sweep = OmegaConf.load(cli_args.sweep_config)
    list_of_sweeps = base_sweep.pop("sweep")
    config_list = []
    for sweep in list_of_sweeps:
        sweep_config = OmegaConf.merge(base_sweep, sweep)
        config_list += grid_to_list(sweep_config)
    overrides = config_list[slurm_task_id - 1]

    if "debug" in cli_args:
        print(len(config_list), config_list)

    run_name = f"{slurm_job_id}-{slurm_task_id}"

    launch_args = [
        f"torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "train_gpt2_sweep.py",
        f"wandb_run_name={run_name}",
    ]

    for k, v in overrides.items():
        launch_args.append(f"{k}={v}")

    if "debug" in cli_args:
        print(launch_args)

    subprocess.run(launch_args)


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    run(cli_args)
