from collections import deque
import numpy as np
from scipy.special import expit

class Traveler:

    def __init__(self, agent_id, connections, pref_av, alpha, wom_coef, inertia, tau, initial_tt_h, initial_tt_av):
        assert 0 <= alpha <= 1
        assert tau > 0
        assert 0 <= wom_coef <= 1
        assert inertia >= 0
        assert all(w >= 0 for w in connections.values())
        assert not connections or np.isclose(sum(connections.values()), 1.0)
        self.id = agent_id
        self.pref_av = pref_av
        self.alpha = alpha
        self.wom_coef = wom_coef
        self.connections = connections
        self.last_choice = 0
        self.cost = np.array([initial_tt_h, initial_tt_av], dtype=float)
        self.inertia = inertia
        self.wom_signal = 0.0
        self.tried_av = False
        self.tau = tau

    def choose(self) -> int:  
        u_h = self.get_utility(0)
        u_av = self.get_utility(1)

        p_av = expit((u_av - u_h) / self.tau)
        choice = int(np.random.random() < p_av)
        self.last_choice = choice
        if not self.tried_av and choice == 1:
            self.tried_av = True
        return choice
        
    def get_utility(self, mode: int) -> float:
        if mode == 0:
            return -self.cost[0] + self.inertia*(self.last_choice == 0)
        elif mode == 1:
            return -self.cost[1] + self.inertia*(self.last_choice == 1) + self.pref_av + self.wom_coef*self.wom_signal
        else:
            raise ValueError("Mode must be 0 (conventional) or 1 (autonomous).")
        
    def learn(self, travel_time: float) -> None:
        self.cost[self.last_choice] = (1 - self.alpha) * self.cost[self.last_choice] + self.alpha * travel_time
    
    def digest_wom(self, woms: dict) -> None:
        if not self.connections: return
        self.wom_signal = 0.0
        for agent_id in self.connections:
            self.wom_signal += self.connections[agent_id] * woms[agent_id]
    
    def emit_wom(self) -> float:
        if self.tried_av:
            av_adv = (self.cost[0] - self.cost[1]) / (self.cost[0] + 1e-14)
            av_adv = np.clip(av_adv, -1, 1)
            return av_adv
        else:
            return 0.0