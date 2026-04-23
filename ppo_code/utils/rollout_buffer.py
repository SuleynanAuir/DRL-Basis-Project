class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int):
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps,), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps,), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.float32)
        self.values = np.zeros((rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((rollout_steps,), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps,), dtype=np.float32)
        self.step = 0
        self.max_steps = rollout_steps

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        index = self.step
        self.obs[index] = obs
        self.actions[index] = action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.dones[index] = float(done)
        self.values[index] = value
        self.step += 1

    def compute_returns_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - float(last_done)
        for index in reversed(range(self.max_steps)):
            delta = (
                self.rewards[index]
                + gamma * next_value * next_non_terminal
                - self.values[index]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[index] = gae
            self.returns[index] = gae + self.values[index]
            next_value = self.values[index]
            next_non_terminal = 1.0 - self.dones[index]

    def get_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        advantages = self.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
            "old_values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
        }
