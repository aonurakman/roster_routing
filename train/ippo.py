import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch

from routerl         import TrafficEnvironment
from routerl         import Keychain as kc
from tqdm            import tqdm
from numpy.random    import normal


from learners        import PPO
from utils           import *
from mode_choice      import Traveler
from components import ClusteredRoutesLoader, resolve_route_set
from components.population import set_daily_population
from components.social_connections import generate_social_network

if __name__ == "__main__":
    parser = add_arguments()
    args = parser.parse_args()
    ALGORITHM = "ippo"
    configure(args.env_seed, args.torch_seed)
    declare_setup(args, ALGORITHM)
    device = get_device()
    params = read_parameters(args, ALGORITHM)
    params.pop("deployment_eps", None)

    for key, value in params.items():
        globals()[key] = value
        
    if not use_clustered_routes:
        raise NotImplementedError("This script requires clustered routes.")
    route_set = resolve_route_set(args.net, args.route_set)
    print(f"Route set: {route_set}")

    custom_network_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../networks/{args.net}")
    phases = [1, human_learning_episodes + 1]
    phase_names = ["Human stabilization", "AV training"]
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    records_folder = os.path.join(base_dir, f"{args.id}_train")
    plots_folder = os.path.join(base_dir, f"{args.id}/plots")

    od_file_path = os.path.join(custom_network_folder, f"od_{args.net}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
        max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}. Please check the network folder.")
            
    total_episodes = human_learning_episodes + training_eps
            
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    if args.project is not None:
        dump_config["project"] = args.project

    action_masks = None
    route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
    clustered_loader = ClusteredRoutesLoader(args.net, route_set_dir=route_set_dir)
    number_of_paths = clustered_loader.get_number_of_paths()
    clustered_loader.export_paths_routes(records_folder, origins, destinations)
    action_masks = clustered_loader.create_masks(origins, destinations)
    if not action_masks:
        raise ValueError("The clustered route set contains no action masks.")

    for od_pair, mask in action_masks.items():
        mask_array = np.asarray(mask)
        if mask_array.shape != (number_of_paths,):
            raise ValueError(
                f"Action mask for OD pair {od_pair} has shape "
                f"{mask_array.shape}; expected ({number_of_paths},)."
            )
        if not np.isin(mask_array, (0, 1)).all():
            raise ValueError(
                f"Action mask for OD pair {od_pair} must be binary."
            )
        if not mask_array.any():
            raise ValueError(
                f"Action mask for OD pair {od_pair} has no valid actions."
            )

    agent_ods = {
        (int(row.origin), int(row.destination))
        for row in pd.read_csv(
            agents_csv_path,
            usecols=["origin", "destination"],
        ).itertuples()
    }
    missing_ods = sorted(agent_ods.difference(action_masks))
    if missing_ods:
        raise ValueError(
            "Missing action masks for agent OD pairs: "
            + ", ".join(map(str, missing_ods))
        )

    create_paths_flag = False
    dump_config["number_of_paths"] = number_of_paths
    dump_config["network"] = args.net
    dump_config["env_seed"] = args.env_seed
    dump_config["torch_seed"] = args.torch_seed
    dump_config["route_set"] = route_set
    dump_config["env_config"] = args.env_conf
    dump_config["task_config"] = args.task_conf
    dump_config["alg_config"] = args.alg_conf
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["use_clustered_routes"] = use_clustered_routes
    dump_config["use_action_masks"] = action_masks is not None
    dump_config["observation_type"] = observations
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = args.env_seed,
        create_agents = False,
        create_paths = False,
        action_masks = action_masks,
        save_detectors_info = False,
        agent_parameters = {
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : observations
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : args.net,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : max_start_time
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    human_by_id = {int(agent.id): agent for agent in env.human_agents}
    traveler_ids = list(human_by_id)
    
    model_config = dict(
        state_size=env.get_observation_function().OBS_SIZE,
        action_space_size=env.action_space_size,
        batch_size=batch_size, lr=lr, num_epochs=num_epochs,
        num_hidden=num_hidden, widths=widths, clip_eps=clip_eps,
        normalize_advantage=normalize_advantage, entropy_coef=entropy_coef,
    )
    shared_model = PPO(**model_config, device=device)
    routing_spec = clustered_loader.routing_spec(records_folder, observations)
    
    ### Learning phase ###
    env.human_learning = humans_adapt_during_pretraining
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        num_agents_to_switch = random.randint(0, len(traveler_ids))
        num_agents_to_switch = (num_agents_to_switch // granularity) * granularity
        av_ids = set(random.sample(traveler_ids, num_agents_to_switch))
        set_daily_population(env, human_by_id, av_ids)
        env.reset()

        if not env.machine_agents:
            env.step()

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            action = None
            if not (termination or truncation):
                agent = human_by_id[int(agent_id)]
                mask = action_masks[(agent.origin, agent.destination)]
                action = shared_model.act(observation, agent_id, mask=mask)
            else:
                shared_model.push(agent_id, reward)
                action = None
                
            env.step(action)

        assert not shared_model.pending, "Some AV actions did not receive rewards"
            
        if episode % update_every == 0:
            shared_model.learn()
            
        if episode % plot_every == 0:
            env.plot_results()
            
        if episode+1 == training_eps or should_save_model_snapshot(episode+1, training_eps, args.save_model_every):
            torch.save(
                {
                    "training_episode": episode+1,
                    "model_config": model_config,
                    "routing_spec": routing_spec,
                    "models": {
                        "policy_net": shared_model.policy_net.state_dict()
                    },
                },
                model_snapshot_path(records_folder, episode+1, "pt"),
            )
        pbar.update()
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()

    loss_records = [
        {"iteration": iteration, "loss": loss}
        for iteration, loss in enumerate(shared_model.loss, start=1)
    ]
    save_loss_records(records_folder, loss_records, columns=["iteration", "loss"])

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(args.id, results_folder="../results")