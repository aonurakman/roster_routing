import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def add_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    parser.add_argument('--training-run', type=str, required=True, help='Training run folder under results')
    parser.add_argument('--action-selection', choices=['sample', 'argmax'], required=True)
    return parser
    
    
def declare_setup(args: argparse.Namespace, alg: str, network: str, route_set: str) -> None:
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {alg.upper()}")
    print(f"Experiment ID: {args.id}")
    print(f"Training run: {args.training_run}")
    print(f"Network: {network}")
    print(f"Environment seed: {args.env_seed}")
    print(f"Algorithm config: {args.alg_conf}")
    print(f"Environment config: {args.env_conf}")
    print(f"Task config: {args.task_conf}")
    print(f"Route set: {route_set}")
    print(f"Metrics will {'NOT ' if args.skip_metrics else ''}be computed after the experiment.\n")
    

def configure(env_seed: int, torch_seed: int) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)
    
def get_device(declare: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    if declare:
        print(f"Device: {device}")
    return device
    
def read_parameters(args: argparse.Namespace, alg: str) -> dict:
    config_root = Path(__file__).resolve().parents[1] / "config"
    params = dict()
    with (config_root / "algo_config" / alg / f"{args.alg_conf}.json").open(encoding="utf-8") as f:
        params.update(json.load(f))
    with (config_root / "env_config" / f"{args.env_conf}.json").open(encoding="utf-8") as f:
        params.update(json.load(f))
    with (config_root / "task_config" / f"{args.task_conf}.json").open(encoding="utf-8") as f:
        params.update(json.load(f))
    return params


def model_snapshot_path(
    records_folder: str,
    completed_episode: int,
    extension: str,
) -> str:
    """Return a standard path for one whole-population policy snapshot."""
    return os.path.join(
        records_folder, "models",
        f"training_ep_{completed_episode:06d}.{extension.lstrip('.')}",
    )


class CSVLossLogger:

    def __init__(self, path: str, columns: list[str]):
        self.path = path
        self.columns = columns

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns, extrasaction='ignore')
        self.writer.writeheader()

    def __call__(self, record: dict)->None:
        self.writer.writerow({column: record.get(column, "") for column in self.columns})
        self.file.flush() # push buffered data to OS immediately

    def close(self):
        # Development note: instead of manual closing, context manager may be added to handle file opening/closing safely using 'with open(...)'
        self.file.close()

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")

    return sorted(eps)


def clear_SUMO_files(sumo_path, ep_path, remove_additional_files=False):
    '''
        Clear SUMO files that are empty or not in the episodes folder.
        Works only for the consecutive files with the same name.
        The files are named as <file_name>_<episode>.xml

        This is a destructive function, it will remove files from the directory!
    '''
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # Ignore a connection that was opened but reset before any vehicle departed.
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("inserted") == "0":
                # remove the file
                os.remove(file_path)
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    
                    
def print_agent_counts(env):
    print(f"""
    ----------------------------------------------------
                    Agents in traffic
    ----------------------------------------------------
    Total agents           | {len(env.all_agents)}
    Human agents           | {len(env.human_agents)}
    AV agents              | {len(env.machine_agents)}
    ----------------------------------------------------
    """)


def script_path_for_config(script_file: str, repo_root: Optional[str] = None) -> str:
    """
    Return a repository-relative script path for exp_config.json when possible.
    Falls back to absolute path if the script is outside of the repository root.
    """
    script_abs = os.path.abspath(script_file)
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = os.path.abspath(repo_root)

    try:
        relative_path = os.path.relpath(script_abs, repo_root)
    except ValueError:
        return script_abs

    if relative_path.startswith(".."):
        return script_abs

    return relative_path.replace(os.sep, "/")


def run_metrics_analysis(exp_id: str, results_folder: str = "../results", verbose: bool = False) -> bool:
    """
    Run analysis/metrics.py for a finished experiment.

    The helper is intentionally non-failing so experiment scripts do not crash
    if post-processing fails.
    """
    metrics_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "analysis", "metrics.py")
    )
    results_folder_path = os.path.abspath(results_folder)

    command = [
        sys.executable,
        metrics_script,
        "--id",
        exp_id,
        "--results-folder",
        results_folder_path,
    ]
    if verbose:
        command.extend(["--verbose", "True"])

    print(f"Running metrics analysis for experiment '{exp_id}'...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"Warning: Metrics analysis failed for experiment '{exp_id}' "
            f"(exit code {result.returncode})."
        )
        return False
    return True
