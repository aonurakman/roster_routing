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
from routerl         import MachineAgent, HumanAgent
from tqdm            import tqdm
from numpy.random    import normal


from learners        import PPO
from utils           import *
from mode_choice      import Traveler
from components import ClusteredRoutesLoader, resolve_route_set
from components.social_connections import generate_social_network

if __name__ == "__main__":
    parser = add_arguments()
    args = parser.parse_args()
    ALGORITHM = "ippo"
    configure(args.env_seed, args.torch_seed)
    declare_setup(args, ALGORITHM)
    device = get_device()
    params = read_parameters(args)

    for key, value in params.items():
        globals()[key] = value
        
    route_set = resolve_route_set(args.net, args.route_set) if use_clustered_routes else None
    print(f"Route set: {route_set or 'none (unclustered)'}")

    custom_network_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../networks/{args.net}")
    phases = [1, human_learning_episodes]
    phase_names = ["Human stabilization", "Deployment"]
    
    base_dir = args.dir if args.dir is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    records_folder = os.path.join(base_dir, f"{args.id}_deploy")
    plots_folder = os.path.join(base_dir, f"{args.id}/plots")
    mode_choice_save_path = os.path.join(records_folder, "mode_choice/")
    os.makedirs(mode_choice_save_path, exist_ok=True)

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
    clustered_loader = ClusteredRoutesLoader(
        args.net,
        custom_network_folder,
        args.shuffle,
        args.env_seed,
        route_set_dir=route_set_dir,
    )
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
    dump_config["shuffle"] = args.shuffle
    dump_config["observation_type"] = observations
    dump_config["path_gen_workers"] = path_gen_workers_value
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

    human_agents_copy = {str(agent.id): agent for agent in env.human_agents}
    
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    # load saved model
    shared_model = PPO(...)
    # make it deterministic for deployment
    shared_model.eval()
    
    mode_choice_model = dict()
    pref_av = normal(loc=pref_av_mean, scale=pref_av_std, size=num_agents)
    wom_coef = normal(loc=wom_coef_mean, scale=wom_coef_std, size=num_agents)
    inertia = normal(loc=inertia_mean, scale=inertia_std, size=num_agents)
    tau = normal(loc=tau_mean, scale=tau_std, size=num_agents)
    av_bias = normal(loc=av_bias_mean, scale=av_bias_std, size=num_agents)
    connections = generate_social_network(env.all_agents, 
                                          topology=social_topology, 
                                          mean_degree=mean_degree,
                                          rewiring_prob=rewiring_prob, 
                                          weight_sigma=weight_sigma, 
                                          ensure_connected=ensure_connected, 
                                          seed=args.env_seed
                                          )
    for idx, agent in enumerate(env.all_agents):
        initial_tt_h = agent.last_reward 
        initial_tt_av = initial_tt_h + av_bias[idx]
        mode_choice_model[str(agent.id)] = Traveler(
            agent.id, connections[idx], pref_av[idx], 
            mc_alpha, wom_coef[idx], inertia[idx], tau[idx], 
            initial_tt_h, initial_tt_av)
        
    mode_choice_stats = list()
    for agent in env.all_agents:
        mode_choice_stats.append(mode_choice_model[str(agent.id)].get_state(mode="full"))
    df = pd.DataFrame(mode_choice_stats)
    df.to_csv(os.path.join(mode_choice_save_path, f"mode_initial.csv"), index=False)
    
    ### Deployment phase ###
    pbar.set_description("AV deployment")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        
        # Mode choice
        for agent in env.all_agents:
            if agent.kind == "Human":
                human_agents_copy[str(agent.id)] = agent
            else:
                env.machine_agents.remove(agent)
                env.all_agents.remove(agent)
                env.human_agents.append(human_agents_copy[str(agent.id)])
                env.all_agents.append(human_agents_copy[str(agent.id)])
            if mode_choice_model[str(agent.id)].choose():
                new_av = MachineAgent(agent.id, agent.start_time,
                                      agent.origin, agent.destination,
                                      env.agent_params[kc.MACHINE_PARAMETERS], 
                                      env.action_space_size)
                env.machine_agents.append(new_av)
                env.all_agents.append(new_av)
                env.human_agents.remove(human_agents_copy[str(agent.id)])
                env.all_agents.remove(human_agents_copy[str(agent.id)])
                
        if episode % save_every == 0:
            mode_choice_stats = list()
            for agent in env.all_agents:
                mode_choice_stats.append(mode_choice_model[str(agent.id)].get_state(mode="short"))
            df = pd.DataFrame(mode_choice_stats)
            df.to_csv(os.path.join(mode_choice_save_path, f"mode_{episode}.csv"), index=False)
            
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            action = None
            if not (termination or truncation):
                action = shared_model.act(observation)
            else:
                mode_choice_model[str(agent_id)].learn(reward)
                
            env.step(action)
        
        wom = {str(agent.id): mode_choice_model[str(agent.id)].emit_wom() for agent in env.all_agents}
        for agent in env.all_agents:
            mode_choice_model[str(agent.id)].digest_wom(wom)
            
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    if not args.skip_metrics:
        run_metrics_analysis(args.id, results_folder="../results")
