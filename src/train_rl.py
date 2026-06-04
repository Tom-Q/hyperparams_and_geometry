"""Online Q-learning training loop (no experience replay, batch=1)."""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model_mlp import MLP
from .rdm import save_activations_mlp, stimuli_to_tensor
from .utils import log4_checkpoints, perf_checkpoint_thresholds, make_optimizer, l1_penalty

EPSILON       = 0.1   # default ε (used when no decay schedule given)
ROLLING_N     = 30    # episodes in rolling average for performance tracking
LOG_INTERVAL  = 5000  # training steps between progress log lines


def train_network(task, config, run_dir, rdm_inputs, env_factory,
                  device=None, max_steps_override=None, verbose=False,
                  epsilon_start=EPSILON, epsilon_end=0.0, epsilon_decay_steps=None,
                  log_interval=LOG_INTERVAL, rolling_n=ROLLING_N,
                  save_activations=True):
    """Online Q-learning. Returns best rolling-average episode return (float)."""
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
    gamma            = 0.99
    stimuli_t        = stimuli_to_tensor(rdm_inputs)
    checkpoint_steps = set(log4_checkpoints(max_steps))

    perf_ckpts   = perf_checkpoint_thresholds(task.chance_perf, task.max_metric)
    perf_crossed = set()

    history       = []
    episode_rets  = deque(maxlen=rolling_n)
    t0            = time.time()

    env         = env_factory()
    global_step = 0
    obs, _      = env.reset()
    ep_ret      = 0.0

    best_rolling = float("-inf")

    while global_step < max_steps:
        global_step += 1

        if epsilon_decay_steps is not None and epsilon_decay_steps > 0:
            epsilon = max(epsilon_end,
                         epsilon_start - (epsilon_start - epsilon_end) * global_step / epsilon_decay_steps)
        else:
            epsilon = epsilon_start

        s      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        q_vals = model(s).squeeze(0)

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = q_vals.detach().argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done    = terminated or truncated
        ep_ret += reward

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

        if save_activations and global_step in checkpoint_steps:
            save_activations_mlp(model, stimuli_t,
                                 run_dir / f"step_{global_step:07d}", device)

        if done:
            episode_rets.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

            if len(episode_rets) == rolling_n:
                rolling_mean = float(np.mean(episode_rets))
                if rolling_mean > best_rolling:
                    best_rolling = rolling_mean
                    if save_activations:
                        for raw_t, label in perf_ckpts:
                            if label not in perf_crossed and best_rolling >= raw_t:
                                save_activations_mlp(model, stimuli_t,
                                                     run_dir / f"perf_{label}", device)
                                perf_crossed.add(label)
                if rolling_mean >= task.success_threshold:
                    if verbose:
                        elapsed = round(time.time() - t0)
                        print(f"  t={elapsed:4d}s  step {global_step:8,}  "
                              f"rolling_mean={rolling_mean:7.3f}  ε={epsilon:.3f}  *** SOLVED ***",
                              flush=True)
                    break
        else:
            obs = next_obs

        if verbose and global_step % log_interval == 0:
            elapsed = round(time.time() - t0)
            rolling_mean = float(np.mean(episode_rets)) if episode_rets else float("nan")
            print(f"  t={elapsed:4d}s  step {global_step:8,}  "
                  f"rolling_mean={rolling_mean:7.3f}  ε={epsilon:.3f}",
                  flush=True)
            if episode_rets:
                history.append({"step": global_step,
                                 "rolling_mean": round(rolling_mean, 4)})

    env.close()

    if save_activations:
        save_activations_mlp(model, stimuli_t, run_dir / "final", device)

    final_rolling = float(np.mean(episode_rets)) if episode_rets else float("-inf")
    if final_rolling > best_rolling:
        best_rolling = final_rolling

    metadata = {
        "task":          task.name,
        "paradigm":      task.paradigm,
        "config":        dict(config),
        "final_step":    global_step,
        "final_rolling": round(final_rolling, 4),
        "best_rolling":  round(best_rolling, 4),
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return float(best_rolling)
