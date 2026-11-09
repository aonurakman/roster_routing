import argparse
import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Optional

from legacy.scripts.baselines_clusters import ALGORITHM


def add_arguments() -> argparse.ArgumentParser:
    # what is the best way of assigning each of these to a variable?
    # answer: use the argparse module to parse command line arguments and assign them to variables
    # show example
    # example: my_variable = args.my_variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--env-conf', type=str, default="clusters")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--route-set', type=str, default=None)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument('--skip-metrics', action='store_true', default=False)
    return parser
    
    
def declare_setup(args: argparse.Namespace, alg: str) -> None:
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {alg.upper()}")
    print(f"Experiment ID: {args.id}")
    print(f"Network: {args.net}")
    print(f"Environment seed: {args.env_seed}")
    print(f"Algorithm config: {args.alg_conf}")
    print(f"Environment config: {args.env_conf}")
    print(f"Task config: {args.task_conf}")
    print(f"Requested route set: {args.route_set or 'network default'}")
    print(f"Shuffle: {args.shuffle}")
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
    
def read_parameters(args: argparse.Namespace) -> dict:
    params = dict()
    params.update(json.load(open(f"../config/algo_config/{ALGORITHM}/{args.alg_conf}.json")))
    params.update(json.load(open(f"../config/env_config/{args.env_conf}.json")))
    params.update(json.load(open(f"../config/task_config/{args.task_conf}.json")))
    return params


def should_save_model_snapshot(
    completed_episode: int,
    training_episodes: int,
    save_model_every: Optional[int],
) -> bool:
    """Return whether a policy snapshot is due after this training episode."""
    return save_model_every is not None and (
        completed_episode % save_model_every == 0
        or completed_episode == training_episodes
    )


def model_snapshot_path(
    records_folder: str,
    completed_episode: int,
    extension: str,
) -> str:
    """Return a standard path for one whole-population policy snapshot."""
    models_folder = os.path.join(records_folder, "models")
    os.makedirs(models_folder, exist_ok=True)
    return os.path.join(
        models_folder,
        f"training_ep_{completed_episode:06d}.{extension.lstrip('.')}",
    )


def warn_model_snapshots_unsupported(save_model_every: Optional[int]) -> None:
    """Explain when a script accepts but does not implement policy snapshots."""
    if save_model_every is not None:
        print("Warning: --save-model-every is not supported by this script and will be ignored.")


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


def save_loss_records(records_folder: str, loss_records: list[dict], columns: list[str]) -> str:
    """
    Save training loss records to a unified CSV file.

    Args:
        records_folder (str): Experiment output folder (e.g. ../results/<exp_id>).
        loss_records (list[dict]): Row-wise loss data.
        columns (list[str]): Ordered CSV columns.

    Returns:
        str: Absolute path to the saved CSV file.
    """
    losses_folder = os.path.join(records_folder, "losses")
    os.makedirs(losses_folder, exist_ok=True)
    loss_csv_path = os.path.join(losses_folder, "losses.csv")

    with open(loss_csv_path, "w", newline="", encoding="utf-8") as loss_file:
        writer = csv.DictWriter(loss_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in loss_records:
            writer.writerow({column: row.get(column, "") for column in columns})

    return os.path.abspath(loss_csv_path)


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
