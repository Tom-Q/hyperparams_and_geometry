"""Q*bert A2C training.

Based directly on atari_acc/ai_acc/base_model.py with minimal additions:
  - gymnasium 1.3 API fixes (FrameStackObservation, register_envs)
  - TensorBoard removed (not a project dependency)
  - _save_activations(): saves perception_fc outputs over stimuli as layer_0 npz
  - evaluate() saves best_weights.pt + best.npz to run_dir
  - train() saves step_NNNNNNN.npz checkpoints, final.npz, metadata.json; returns best score
  - train_network() wrapper for use by scripts/run_qbert_network.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


STEP_CHECKPOINTS    = frozenset({1, 4, 16, 64, 256, 1024, 2048, 3072, 4096, 5120, 6144, 7168})
QBERT_LEVEL_RAM_ADDR = 99   # RAM byte storing cumulative levels-completed count (addr 57 wraps at level 5)
TRACKED_LEVELS       = tuple(range(2, 21))

SUCCESS_LEVEL  = 5          # level that must be reached to declare success
SUCCESS_FRAC   = 0.5        # fraction of eval episodes that must reach SUCCESS_LEVEL
PATIENCE_STEPS = 10_000_000 # stop if eval_mean hasn't improved in this many env steps

PERF_THRESHOLDS = {
    0.025: "0p025",
    0.05:  "0p05",
    0.1:   "0p1",
    0.2:   "0p2",
    0.4:   "0p4",
    0.6:   "0p6",
    0.8:   "0p8",
    0.85:  "0p85",
    0.9:   "0p9",
    0.95:  "0p95",
}

MAX_METRIC  = 15_000.0
CHANCE_PERF = 0.0


def _register_ale():
    import ale_py
    gym.register_envs(ale_py)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        if self.use_batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += residual
        return F.relu(out)


class SpatialAttentionModule(nn.Module):
    """Multi-head self-attention over the 7×7 spatial grid of conv features.

    Treats each of the 49 spatial positions as a token (64-dim), adds learnable
    positional embeddings so positions are distinguishable, then applies MHSA
    with a pre-norm + residual pattern before reshaping back to (B, C, H, W).
    """
    def __init__(self, channels=64, num_heads=4, grid_size=7):
        super().__init__()
        n_tokens = grid_size * grid_size  # 49

        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, channels))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 64)
        tokens = tokens + self.pos_embedding
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out                       # residual
        return tokens.permute(0, 2, 1).view(B, C, H, W)


class ImprovedA2CNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, use_attention=False, use_residual=False,
                 use_aux=False, use_batch_norm=False, hidden_size=512, depth=1):
        super(ImprovedA2CNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_aux = use_aux
        self.use_batch_norm = use_batch_norm
        self.hidden_size = hidden_size
        self.depth = depth

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        if use_batch_norm:
            self.bn1 = nn.GroupNorm(8, 32)
            self.bn2 = nn.GroupNorm(16, 64)
            self.bn3 = nn.GroupNorm(16, 64)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        if self.use_residual:
            self.residual = ResidualBlock(64, use_batch_norm=use_batch_norm)

        if self.use_attention:
            self.attention = SpatialAttentionModule(channels=64, num_heads=4, grid_size=7)

        conv_out_size = 3136

        self.perception_fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity(),
            nn.Tanh()
        )

        self.fc2 = (nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity(),
            nn.Tanh()
        ) if depth >= 2 else None)

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        if use_aux:
            self.next_frame_predictor = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.GELU(),
                nn.Linear(512, np.prod(input_shape))
            )
            self.reward_predictor = nn.Sequential(
                nn.Linear(conv_out_size, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            policy_modules = [m for m in self.policy_net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            value_modules  = [m for m in self.value_net.modules()  if isinstance(m, (nn.Conv2d, nn.Linear))]
            perception_modules = [m for m in self.perception_fc.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            if self.fc2 is not None:
                perception_modules += [m for m in self.fc2.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

            if module in policy_modules:
                gain = 0.01 if module is policy_modules[-1] else 5.0/3
            elif module in value_modules:
                gain = 5.0/3
            elif module in perception_modules:
                gain = 1.7724
            else:
                gain = np.sqrt(2.0)

            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, return_activations=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if self.use_residual:
            x = self.residual(x)
        if self.use_attention:
            x = self.attention(x)

        features = x.view(x.size(0), -1)
        layer0   = self.perception_fc(features)
        layer1   = self.fc2(layer0) if self.fc2 is not None else None
        out      = layer1 if layer1 is not None else layer0

        if self.use_aux:
            next_frame       = self.next_frame_predictor(features)
            predicted_reward = self.reward_predictor(features)
        else:
            next_frame = predicted_reward = None

        action_logits        = self.policy_net(out)
        action_probabilities = F.softmax(action_logits, dim=1)
        value                = self.value_net(out)

        return action_probabilities, value, features, next_frame, predicted_reward

    def get_action_and_value(self, x, deterministic=False):
        policy, value, _, _, _ = self(x)
        if deterministic:
            action = torch.argmax(policy, dim=1)
        else:
            dist   = Categorical(policy)
            action = dist.sample()
        action_log_probs = torch.log(policy + 1e-10).gather(1, action.unsqueeze(1))
        entropy = -(policy * torch.log(policy + 1e-10)).sum(1, keepdim=True)
        return action, action_log_probs, value, entropy


class PPOBuffer:
    def __init__(self, state_shape, action_shape, n_steps, n_envs, gamma=0.99, lam=0.95, device='cuda'):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.device  = device
        self.gamma   = gamma
        self.lam     = lam
        self.step    = 0

        self.states           = torch.zeros((n_steps, n_envs, *state_shape),  dtype=torch.float32, device=device)
        self.actions          = torch.zeros((n_steps, n_envs, *action_shape), dtype=torch.long,    device=device)
        self.rewards          = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.intrinsic_rewards= torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.values           = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.log_probs        = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.advantages       = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.returns          = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)
        self.next_states      = torch.zeros((n_steps, n_envs, *state_shape),  dtype=torch.float32, device=device)
        self.masks            = torch.zeros((n_steps, n_envs, 1),             dtype=torch.float32, device=device)

    def add(self, states, actions, rewards, intrinsic_rewards, values, log_probs, next_states, masks):
        if self.step >= self.n_steps:
            print("WARNING: Buffer capacity exceeded.")
            return
        self.states[self.step]            = states
        self.actions[self.step]           = actions
        self.rewards[self.step]           = rewards
        self.intrinsic_rewards[self.step] = intrinsic_rewards
        self.values[self.step]            = values
        self.log_probs[self.step]         = log_probs
        self.next_states[self.step]       = next_states
        self.masks[self.step]             = masks
        self.step += 1

    def compute_advantages(self, last_value, use_intrinsic=True):
        if self.step == 0:
            return
        gae        = torch.zeros((self.n_envs, 1), device=self.device)
        next_value = last_value
        for t in reversed(range(self.step)):
            current_rewards = (self.rewards[t] + 0.01 * self.intrinsic_rewards[t]
                               if use_intrinsic else self.rewards[t])
            delta      = current_rewards + self.gamma * next_value * self.masks[t] - self.values[t]
            gae        = delta + self.gamma * self.lam * self.masks[t] * gae
            self.advantages[t] = gae
            next_value = self.values[t]
        self.returns = self.advantages + self.values

    def get_all(self):
        states         = self.states[:self.step].reshape(-1, *self.states.shape[2:])
        actions        = self.actions[:self.step].reshape(-1, *self.actions.shape[2:])
        rewards        = self.rewards[:self.step].reshape(-1, 1)
        intrinsic_rewards = self.intrinsic_rewards[:self.step].reshape(-1, 1)
        values         = self.values[:self.step].reshape(-1, 1)
        log_probs      = self.log_probs[:self.step].reshape(-1, 1)
        advantages     = self.advantages[:self.step].reshape(-1, 1)
        returns        = self.returns[:self.step].reshape(-1, 1)
        next_states    = self.next_states[:self.step].reshape(-1, *self.next_states.shape[2:])
        masks          = self.masks[:self.step].reshape(-1, 1)
        return {
            "states": states, "actions": actions, "rewards": rewards,
            "intrinsic_rewards": intrinsic_rewards, "values": values,
            "log_probs": log_probs, "advantages": advantages, "returns": returns,
            "next_states": next_states, "masks": masks,
        }

    def clear(self):
        self.step = 0


class ImprovedA2CAgent:

    def __init__(
            self,
            env_name: str,
            num_envs: int = 16,
            learning_rate: float = 2.5e-4,
            gamma: float = 0.99,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            aux_coef: float = 0.1,
            clip_grad_norm: float = 0.5,
            n_steps: int = 512,
            update_epochs: int = 2,
            batch_size: int = 256,
            clip_ratio: float = 0.2,
            use_rnd: bool = False,
            use_attention: bool = False,
            use_residual: bool = False,
            use_aux: bool = False,
            use_batch_norm: bool = False,
            use_ppo: bool = True,
            hidden_size: int = 512,
            depth: int = 1,
            save_name="ImprovedA2CAgent",
            run_dir: Path = None,
            stimuli_t: torch.Tensor = None,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
            self.env_name      = env_name
            self.num_envs      = num_envs
            self.learning_rate = learning_rate
            self.device        = device
            self.gamma         = gamma
            self.entropy_coef  = entropy_coef
            self.value_coef    = value_coef
            self.aux_coef      = aux_coef
            self.clip_grad_norm= clip_grad_norm
            self.n_steps       = n_steps
            self.update_epochs = update_epochs
            self.batch_size    = batch_size
            self.clip_ratio    = clip_ratio
            self.use_rnd       = use_rnd
            self.use_aux       = use_aux
            self.use_batch_norm= use_batch_norm
            self.use_ppo       = use_ppo
            self.save_name     = save_name
            self.run_dir       = Path(run_dir) if run_dir else Path(".")
            self.stimuli_t     = stimuli_t   # (N, 4, 84, 84) float32 tensor or None

            self.envs       = self._make_envs(env_name, num_envs)
            self.obs_shape  = self.envs.single_observation_space.shape
            self.action_shape = ()
            self.num_actions  = self.envs.single_action_space.n

            self.network = ImprovedA2CNetwork(
                self.obs_shape,
                self.num_actions,
                use_attention  = use_attention,
                use_residual   = use_residual,
                use_aux        = use_aux,
                use_batch_norm = self.use_batch_norm,
                hidden_size    = hidden_size,
                depth          = depth,
            ).to(device)

            if use_rnd:
                dummy_input = torch.zeros(1, *self.obs_shape).to(device)
                _, _, features, _, _ = self.network(dummy_input)
                from .rnd import RND
                self.rnd = RND(features.shape[1]).to(device)
                self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=learning_rate)
            else:
                self.rnd = None

            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-7)

            self.buffer = PPOBuffer(
                self.obs_shape,
                self.action_shape,
                n_steps,
                num_envs,
                gamma  = gamma,
                device = device,
            )

            self.total_steps       = 0
            self.episodes          = 0
            self.best_mean_reward  = -float('inf')
            self.episode_rewards   = [0 for _ in range(num_envs)]
            self.reward_ema              = None  # EMA of raw episode rewards (alpha=0.01 per episode)
            self.perf_thresholds_crossed = set()  # normalised thresholds already saved as perf_X.npz

            self._log_path = run_dir / "training_log.csv"
            level_cols = ",".join(
                f"frac_level{L},steps_to_level{L}_mean,steps_to_level{L}_std"
                for L in TRACKED_LEVELS
            )
            with open(self._log_path, "w") as f:
                f.write(f"step,eval_mean,eval_std,ema_reward,{level_cols}\n")

    def _make_envs(self, env_name, num_envs):
        from gymnasium.vector import SyncVectorEnv
        from gymnasium.wrappers import FrameStackObservation
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

        _register_ale()

        def make_env():
            def _init():
                env = gym.make(env_name, render_mode=None)
                env = AtariPreprocessing(
                    env,
                    noop_max=30,
                    frame_skip=1,
                    screen_size=84,
                    terminal_on_life_loss=True,
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=True,
                )
                env = FrameStackObservation(env, 4)
                return env
            return _init

        return SyncVectorEnv([make_env() for _ in range(num_envs)])

    def _process_rewards(self, rewards):
        return torch.sign(rewards).float()

    def _save_activations(self, path, step=None):
        """Save FC layer activations over stimuli, with step and rolling avg reward."""
        if self.stimuli_t is None:
            return
        net = self.network
        net.eval()
        with torch.no_grad():
            x = self.stimuli_t.to(self.device)
            x = F.relu(net.bn1(net.conv1(x)))
            x = F.relu(net.bn2(net.conv2(x)))
            x = F.relu(net.bn3(net.conv3(x)))
            if net.use_residual:
                x = net.residual(x)
            if net.use_attention:
                x = net.attention(x)
            flat   = x.view(x.size(0), -1)
            layer0 = net.perception_fc(flat)
            layer1 = net.fc2(layer0) if net.fc2 is not None else None

        arrays = dict(
            layer_0    = layer0.cpu().numpy().astype(np.float32),
            step       = np.array(self.total_steps if step is None else step, dtype=np.int64),
            avg_reward = np.array(self.reward_ema if self.reward_ema is not None else np.nan,
                                  dtype=np.float32),
        )
        if layer1 is not None:
            arrays["layer_1"] = layer1.cpu().numpy().astype(np.float32)
        np.savez_compressed(str(path), **arrays)
        net.train()

    def collect_rollouts(self):
        if not hasattr(self, 'current_states'):
            obs, info = self.envs.reset()
            self.current_states = torch.tensor(obs, dtype=torch.float32).to(self.device)
            self.episode_infos = [
                {'reward': 0, 'length': 0, 'completed': False}
                for _ in range(self.num_envs)
            ]
            if 'lives' in info and isinstance(info['lives'], np.ndarray):
                self.prev_lives = info['lives'].copy()
            else:
                self.prev_lives = np.full(self.num_envs, 3)
            self.env_needs_reset = [False] * self.num_envs

        self.buffer.clear()
        total_completed_episodes = 0
        total_episode_returns    = 0.0

        for step in range(self.n_steps):
            if any(self.env_needs_reset):
                for i, needs_reset in enumerate(self.env_needs_reset):
                    if needs_reset:
                        temp_env = self.envs.envs[i]
                        obs, _   = temp_env.reset()
                        self.current_states[i] = torch.tensor(obs, dtype=torch.float32).to(self.device)
                        self.prev_lives[i]     = 3
                        self.env_needs_reset[i]= False

            with torch.no_grad():
                policy, values, features, _, _ = self.network(self.current_states)
                dist      = Categorical(policy)
                actions   = dist.sample()
                log_probs = dist.log_prob(actions).unsqueeze(1)

                if self.use_rnd:
                    intrinsic_rewards = self.rnd(features).detach()
                else:
                    intrinsic_rewards = torch.zeros((self.num_envs, 1), device=self.device)

            current_states_copy = self.current_states.clone()
            cpu_actions         = actions.cpu().numpy()
            next_states, rewards, terminated, truncated, infos = self.envs.step(cpu_actions)
            dones = terminated | truncated

            next_states_tensor  = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            rewards_tensor      = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1).to(self.device)
            processed_rewards   = self._process_rewards(rewards_tensor)
            masks               = torch.tensor(~dones, dtype=torch.float32).reshape(-1, 1).to(self.device)

            if 'lives' in infos and isinstance(infos['lives'], np.ndarray):
                current_lives = infos['lives']
            else:
                current_lives = self.prev_lives.copy()

            for i, (done, reward) in enumerate(zip(dones, rewards_tensor)):
                self.episode_infos[i]['reward'] += reward.item()
                self.episode_infos[i]['length'] += 1
                episode_done = done or current_lives[i] == 0

                if episode_done:
                    if not self.episode_infos[i]['completed']:
                        self.episodes += 1
                        total_completed_episodes += 1
                        ep_reward = self.episode_infos[i]['reward']
                        total_episode_returns += ep_reward
                        self.reward_ema = (ep_reward if self.reward_ema is None
                                          else 0.99 * self.reward_ema + 0.01 * ep_reward)
                        self.episode_infos[i]['completed'] = True
                    self.env_needs_reset[i]  = True
                    self.episode_infos[i]    = {'reward': 0, 'length': 0, 'completed': False}

            self.prev_lives = current_lives.copy()

            self.buffer.add(
                current_states_copy, actions, processed_rewards,
                intrinsic_rewards, values, log_probs, next_states_tensor, masks,
            )
            self.current_states  = next_states_tensor.clone()
            self.total_steps    += self.num_envs

        with torch.no_grad():
            _, last_value, _, _, _ = self.network(self.current_states)

        self.buffer.compute_advantages(last_value)

        avg_return = total_episode_returns / max(1, total_completed_episodes)
        avg_intrinsic_reward = torch.mean(intrinsic_rewards).item() if self.use_rnd else 0.0

        return self.buffer.get_all(), {
            'avg_return':           avg_return,
            'completed_episodes':   total_completed_episodes,
            'avg_intrinsic_reward': avg_intrinsic_reward,
        }

    def update_policy(self, rollout_data, return_losses=False):
        policy_loss_value = value_loss_value = entropy_value = 0

        states     = rollout_data['states']
        actions    = rollout_data['actions']
        old_values = rollout_data['values']
        old_log_probs = rollout_data['log_probs']
        returns    = rollout_data['returns']
        advantages = rollout_data['advantages']
        next_states= rollout_data['next_states']

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            indices = torch.randperm(states.size(0))
            for start in range(0, states.size(0), self.batch_size):
                end         = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states    = states[batch_indices]
                batch_actions   = actions[batch_indices]
                batch_old_lp    = old_log_probs[batch_indices]
                batch_returns   = returns[batch_indices]
                batch_advantages= advantages[batch_indices]
                batch_next      = next_states[batch_indices]

                policy, values, features, predicted_next, predicted_rewards = self.network(batch_states)

                dist       = torch.distributions.Categorical(policy)
                new_log_probs = dist.log_prob(batch_actions.squeeze()).unsqueeze(1)
                entropy    = dist.entropy().mean()

                if self.use_ppo:
                    ratio  = torch.exp(new_log_probs - batch_old_lp)
                    surr1  = ratio * batch_advantages
                    surr2  = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                else:
                    policy_loss = -(new_log_probs * batch_advantages).mean()

                value_loss = F.mse_loss(values, batch_returns)

                policy_loss_value = policy_loss.item()
                value_loss_value  = value_loss.item()
                entropy_value     = entropy.item()

                if self.use_aux:
                    next_state_loss = F.mse_loss(
                        predicted_next,
                        batch_next.view(batch_next.size(0), -1),
                    )
                else:
                    next_state_loss = 0

                if self.use_rnd:
                    intrinsic_rewards = self.rnd(features)
                    with torch.no_grad():
                        target_features = self.rnd.target(features)
                    rnd_loss = F.mse_loss(self.rnd.predictor(features), target_features)
                    self.rnd_optimizer.zero_grad()
                    rnd_loss.backward(retain_graph=True)
                    self.rnd_optimizer.step()
                else:
                    rnd_loss = torch.tensor(0.0, device=self.device)

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + (self.aux_coef * next_state_loss if self.use_aux else 0)
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                self.optimizer.step()

        if return_losses:
            return {'policy_loss': policy_loss_value, 'value_loss': value_loss_value, 'entropy': entropy_value}

    def evaluate(self, num_episodes=10, deterministic=False, verbose=False):
        from gymnasium.wrappers import FrameStackObservation
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

        _register_ale()
        eval_env = gym.make(self.env_name, render_mode=None)
        eval_env = AtariPreprocessing(
            eval_env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True,
        )
        eval_env = FrameStackObservation(eval_env, 4)

        self.network.eval()
        all_rewards = []
        ale = eval_env.unwrapped.ale
        # steps_to_level[L] = list of step counts when level L was first reached, per episode
        steps_to_level = {L: [] for L in TRACKED_LEVELS}

        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            ep_step = 0
            current_level = 1
            done = False

            while not done:
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, _, _, _, _ = self.network(state_tensor)
                    action = (torch.argmax(policy, dim=1) if deterministic
                              else Categorical(policy).sample()).cpu().numpy()[0]
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                ep_step += 1

                new_level = int(ale.getRAM()[QBERT_LEVEL_RAM_ADDR]) + 1
                if new_level > current_level:
                    for L in TRACKED_LEVELS:
                        if L == new_level:
                            steps_to_level[L].append(ep_step)
                    current_level = new_level

                done = terminated or truncated

            all_rewards.append(episode_reward)
            if verbose:
                print(f"Episode {episode+1}/{num_episodes}: {episode_reward:.1f}  max_level={current_level}")

        eval_env.close()

        mean_reward = np.mean(all_rewards)
        std_reward  = np.std(all_rewards)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            torch.save(self.network.state_dict(), self.run_dir / "best_weights.pt")
            self._save_activations(self.run_dir / "best")

        norm_perf = (mean_reward - CHANCE_PERF) / (MAX_METRIC - CHANCE_PERF)
        for thr, name in PERF_THRESHOLDS.items():
            if thr not in self.perf_thresholds_crossed and norm_perf >= thr:
                self.perf_thresholds_crossed.add(thr)
                self._save_activations(self.run_dir / f"perf_{name}")

        ema = self.reward_ema if self.reward_ema is not None else float('nan')
        frac_success_level = len(steps_to_level[SUCCESS_LEVEL]) / num_episodes
        level_vals = []
        for L in TRACKED_LEVELS:
            reached = steps_to_level[L]
            frac = len(reached) / num_episodes
            mean_steps = float(np.mean(reached)) if reached else float('nan')
            std_steps  = float(np.std(reached))  if reached else float('nan')
            level_vals.extend([f"{frac:.3f}", f"{mean_steps:.1f}", f"{std_steps:.1f}"])
        with open(self._log_path, "a") as f:
            f.write(f"{self.total_steps},{mean_reward:.2f},{std_reward:.2f},{ema:.2f},"
                    + ",".join(level_vals) + "\n")

        self.network.train()
        return mean_reward, std_reward, frac_success_level

    def train(self, total_timesteps, eval_freq=200_000, log_freq=10_000):
        rollout_size = self.n_steps * self.num_envs
        num_updates  = total_timesteps // rollout_size
        t0 = time.time()

        best_train_reward = -float('inf')
        avg_train_reward  = 0
        steps_at_last_improvement = None  # set on first eval (which always improves from -inf)
        stop_reason = "completed"

        for update in range(num_updates):
            rollout_data, train_info = self.collect_rollouts()

            if train_info['completed_episodes'] > 0:
                avg_train_reward = train_info['avg_return']
                best_train_reward = max(best_train_reward, avg_train_reward)

            self.update_policy(rollout_data)

            log_update = update % max(1, log_freq // rollout_size) == 0
            if log_update:
                elapsed = round(time.time() - t0)
                print(f"  t={elapsed:5d}s  update={update}/{num_updates}  "
                      f"steps={self.total_steps:>12,}  avg_ep={avg_train_reward:.1f}", flush=True)

            if self.total_steps % eval_freq < rollout_size:
                prev_best = self.best_mean_reward
                mean_reward, std_reward, frac_success = self.evaluate()
                print(f"  [eval] steps={self.total_steps:,}  "
                      f"score={mean_reward:.1f}±{std_reward:.1f}  "
                      f"best={self.best_mean_reward:.1f}  "
                      f"frac_level{SUCCESS_LEVEL}={frac_success:.2f}", flush=True)

                if self.best_mean_reward > prev_best:
                    steps_at_last_improvement = self.total_steps

                if frac_success >= SUCCESS_FRAC:
                    print(f"  [early stop] SUCCESS: level {SUCCESS_LEVEL} reached in "
                          f"{frac_success:.0%} of episodes at step {self.total_steps:,}", flush=True)
                    stop_reason = "success"
                    break

                if (steps_at_last_improvement is not None and
                        self.total_steps - steps_at_last_improvement >= PATIENCE_STEPS):
                    print(f"  [early stop] PATIENCE: no improvement for "
                          f"{self.total_steps - steps_at_last_improvement:,} steps", flush=True)
                    stop_reason = "patience"
                    break

            update_count = update + 1  # 1-indexed
            if update_count in STEP_CHECKPOINTS:
                self._save_activations(self.run_dir / f"step_{update_count:07d}", step=self.total_steps)

        # Final evaluation and checkpoint (always runs, even on early stop)
        mean_reward, std_reward, _ = self.evaluate(num_episodes=30)
        print(f"  [final eval] score={mean_reward:.1f}±{std_reward:.1f}  "
              f"stop_reason={stop_reason}", flush=True)
        self._save_activations(self.run_dir / "final")
        self.envs.close()

        return float(self.best_mean_reward), stop_reason


def train_network(config, run_dir, stimuli_array, total_steps=60_000_000,
                  n_envs=16, device=None, verbose=True):
    """Train one Q*bert network. Returns best evaluation score."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stimuli_t = torch.tensor(stimuli_array, dtype=torch.float32)

    agent = ImprovedA2CAgent(
        env_name       = "ALE/Qbert-v5",
        num_envs       = n_envs,
        learning_rate  = float(config["learning_rate"]),
        gamma          = float(config["gamma"]),
        entropy_coef   = float(config["entropy_coef"]),
        use_batch_norm = bool(config.get("use_batch_norm", True)),
        use_attention  = bool(config.get("use_attention",  False)),
        use_residual   = bool(config.get("use_residual",   False)),
        use_ppo        = True,
        hidden_size    = int(config.get("hidden_size", 512)),
        depth          = int(config.get("depth", 1)),
        run_dir        = run_dir,
        stimuli_t      = stimuli_t,
        device         = device,
    )

    t0 = time.time()
    best_score, stop_reason = agent.train(total_timesteps=total_steps)

    metadata = {
        "task":     "qbert",
        "paradigm": "qbert",
        "config": {
            **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in config.items()},
        },
        "best_metric":     round(best_score, 2),
        "final_step":      agent.total_steps,
        "stop_reason":     stop_reason,
        "training_time_s": round(time.time() - t0, 1),
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  Done. best={best_score:.1f}  steps={agent.total_steps:,}  "
              f"t={round(time.time()-t0)}s", flush=True)

    return best_score
