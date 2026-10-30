import random
import torch

from torch import optim
from torch import nn

from .common import Network

class PPO:
    def __init__(self, state_size, action_space_size,
                 device="cpu", batch_size=16, lr=0.003, num_epochs=4,
                 num_hidden=2, widths=[32, 64, 32], clip_eps=0.2,
                 normalize_advantage=True, entropy_coef=0.3,
                 action_mask=None):
        super().__init__()
        self.device = device
        self.action_space_size = action_space_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef

        self.policy_net = Network(state_size, action_space_size, num_hidden, widths).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=-1)
        
        self.loss = list()
        self.memory = list()
        self.deterministic = False

        if action_mask is None:
            self.action_mask = None
        else:
            self.action_mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

    def _distribution(self, state_tensor, mask=None):
        logits = self.policy_net(state_tensor)
        if self.action_mask is not None:
            logits = logits.masked_fill(
                ~self.action_mask.unsqueeze(0),
                float("-inf"),
            )
        elif mask is not None:
            mask_tensor = torch.as_tensor(
                mask,
                dtype=torch.bool,
                device=self.device,
            )
            logits = logits.masked_fill(
                ~mask_tensor.unsqueeze(0),
                float("-inf"),
            )
        return torch.distributions.Categorical(probs=self.softmax(logits))

    def act(self, state, mask=None):
        if mask is not None and self.action_mask is not None:
            raise ValueError("Cannot provide both a step mask and a persistent action mask.")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self._distribution(state_tensor, mask)
        if not self.deterministic:
            action = dist.sample().item()
        else:
            action = torch.argmax(dist.probs).item()
        self.last_state = state
        self.last_action = action
        self.last_log_prob = dist.log_prob(
            torch.tensor(action, device=self.device)
        ).item()
        return action

    def push(self, reward):
        self.memory.append((self.last_state, self.last_action, self.last_log_prob, reward))
        del self.last_state, self.last_action, self.last_log_prob

    def learn(self):
        if len(self.memory) < self.batch_size: return
        step_loss = list()

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards = zip(*batch)
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)

            dist = self._distribution(states_tensor)
            new_log_probs = dist.log_prob(actions_tensor)

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            if self.normalize_advantage: advantage = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            else: advantage = rewards_tensor

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            step_loss.append(loss.item())

        self.loss.append(sum(step_loss) / len(step_loss))
        self.memory.clear()