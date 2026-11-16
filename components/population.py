from routerl import Keychain as kc, MachineAgent


def set_daily_population(env, human_by_id, av_ids):
    env.human_agents = [agent for agent_id, agent in human_by_id.items() if agent_id not in av_ids]
    env.machine_agents = [
        MachineAgent(agent.id, agent.start_time, agent.origin, agent.destination,
                     env.agent_params[kc.MACHINE_PARAMETERS], env.action_space_size)
        for agent_id, agent in human_by_id.items()
        if agent_id in av_ids
    ]
    env.machine_agents.sort(key=lambda agent: (agent.start_time, agent.id))
    env.all_agents = env.human_agents + env.machine_agents

    # RouteRL reset uses the existing agent IDs and spaces.
    env._initialize_machine_agents()
    env.observation_space.cache_clear()
    env.action_space.cache_clear()
    if not env.machine_agents:
        env.agent_selection = None
        env._agent_selector = None
