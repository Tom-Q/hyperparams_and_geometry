"""Online Q-learning training loop (no experience replay, batch=1)."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model_mlp import MLP
from .rdm import save_activations_mlp, stimuli_to_tensor
from .utils import log4_checkpoints, make_optimizer, l1_penalty

EPSILON       = 0.1    # fixed ε for ε-greedy; not a GP variable
EVAL_INTERVAL = 100    # episodes between mean-return evaluations
EVAL_EPISODES = 100    # episodes averaged for the success check


def _eval_mean_return(model, env_factory, n_episodes, device):
    env   = env_factory()
    total = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            done   = False
            while not done:
                s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                q = model(s).squeeze(0)
                action = q.argmax().item()
                obs, r, terminated, truncated, _ = env.step(action)
                done    = terminated or truncated
                ep_ret += r
            total += ep_ret
    env.close()
    model.train()
    return total / n_episodes


def train_network(task, config, run_dir, rdm_inputs, env_factory,
                  device=None, max_steps_override=None, verbose=False):
    """Online Q-learning. Returns mean_return (float)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_steps = max_steps_override if max_steps_override is not None else task.max_steps

    model = MLP(
        input_size  = task.input_size,
        output_size = task.output_size,
        hidden_size = int(config["hidden_size"]),
        depth       = int(config["depth"]),
        activation  = config["activation"],
        init_scale  = float(config["init_scale"]),
    ).to(device)

    optimizer        = make_optimizer(model, config)
    criterion        = nn.MSELoss()
    l1_coef          = config["l1_reg"]
    gamma            = float(config["gamma"])
    stimuli_t        = stimuli_to_tensor(rdm_inputs)
    checkpoint_steps = set(log4_checkpoints(max_steps))

    env           = env_factory()
    global_step   = 0
    episode_count = 0
    final_return  = 0.0

    curve_steps, curve_returns = [], []

    obs, _ = env.reset()
    ep_ret = 0.0

    while global_step < max_steps:
        global_step += 1

        s      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        q_vals = model(s).squeeze(0)

        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = q_vals.detach().argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        with torch.no_grad():
            s_next   = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            q_next   = model(s_next).squeeze(0)
            target_q = reward + (0.0 if done else gamma * q_next.max().item())

        q_pred = q_vals[action]
        loss   = criterion(q_pred, torch.tensor(target_q, dtype=torch.float32).to(device))
        loss  += l1_penalty(model, l1_coef)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_ret += reward

        if global_step in checkpoint_steps:
            save_activations_mlp(model, stimuli_t, global_step, run_dir, device)

        if done:
            episode_count += 1
            ep_ret         = 0.0
            obs, _         = env.reset()

            if episode_count % EVAL_INTERVAL == 0:
                mean_ret = _eval_mean_return(model, env_factory, EVAL_EPISODES, device)
                final_return = mean_ret
                curve_steps.append(global_step)
                curve_returns.append(mean_ret)
                if verbose:
                    tag = "  *** SOLVED ***" if mean_ret >= task.success_threshold else ""
                    print(f"  ep {episode_count:6d}  step {global_step:8,}  "
                          f"mean_return={mean_ret:7.2f}{tag}", flush=True)
                if mean_ret >= task.success_threshold:
                    break
        else:
            obs = next_obs

    env.close()

    # Final evaluation if training ended without a recent EVAL_INTERVAL check
    if not curve_returns or curve_returns[-1] != final_return:
        final_return = _eval_mean_return(model, env_factory, EVAL_EPISODES, device)
        curve_steps.append(global_step)
        curve_returns.append(final_return)

    np.savez(
        run_dir / "training_curves.npz",
        steps   = np.array(curve_steps),
        returns = np.array(curve_returns),
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump(dict(config), f, indent=2)

    return float(final_return)
