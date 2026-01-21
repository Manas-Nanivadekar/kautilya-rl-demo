"""PPO Actor-Critic agent for signal weighting."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class ActorCritic(nn.Module):
    """Shared backbone with separate actor/critic heads."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor outputs mean weights (will be softmaxed)
        self.actor_mean = nn.Linear(hidden, action_dim)
        # Log std as learnable parameter
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        # Critic outputs state value
        self.critic = nn.Linear(hidden, 1)

        # Initialize with small weights for actor
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        # Actor: output logits, softmax applied later for weights
        action_mean = self.actor_mean(features)
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        # Critic
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        action_mean, action_std, value = self(state)
        if deterministic:
            # Softmax to get weights
            weights = torch.softmax(action_mean, dim=-1)
            return weights, None, value
        # Sample from Gaussian, then softmax
        dist = torch.distributions.Normal(action_mean, action_std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        weights = torch.softmax(action_raw, dim=-1)
        return weights, log_prob, value

    def evaluate_action(self, state: torch.Tensor, action_raw: torch.Tensor):
        action_mean, action_std, value = self(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOAgent:
    """PPO agent with GAE and clipped objective."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Rollout buffer
        self.states: list[np.ndarray] = []
        self.actions_raw: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean, action_std, value = self.model(state_t)
            if deterministic:
                weights = torch.softmax(action_mean, dim=-1)
                return weights.squeeze(0).cpu().numpy()
            dist = torch.distributions.Normal(action_mean, action_std)
            action_raw = dist.sample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            weights = torch.softmax(action_raw, dim=-1)

        # Store for training
        self.states.append(state)
        self.actions_raw.append(action_raw.squeeze(0))
        self.log_probs.append(log_prob.squeeze(0))
        self.values.append(value.squeeze())
        return weights.squeeze(0).cpu().numpy()

    def store_reward(self, reward: float, done: bool = False):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        values = [v.item() for v in self.values] + [next_value]
        gae = 0.0
        advantages = []
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor([v.item() for v in self.values]).to(self.device)
        return advantages, returns

    def update(self, n_epochs: int = 10, batch_size: int = 64) -> dict:
        """PPO update with multiple epochs."""
        if len(self.rewards) == 0:
            return {"loss": 0.0}

        # Compute advantages
        with torch.no_grad():
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            _, _, last_value = self.model(last_state)
            next_val = last_value.item() if not self.dones[-1] else 0.0
        advantages, returns = self.compute_gae(next_val)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_raw = torch.stack(self.actions_raw).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)

        # PPO epochs
        total_loss = 0.0
        n_samples = len(self.rewards)
        indices = np.arange(n_samples)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions_raw[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate current policy
                log_probs, entropy, values = self.model.evaluate_action(batch_states, batch_actions)

                # PPO clipped objective
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        # Clear buffer
        self.states, self.actions_raw, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

        return {"loss": total_loss / (n_epochs * max(1, n_samples // batch_size))}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
