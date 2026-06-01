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
EVAL_INTERVAL = 1000   # environment steps between mean-return evaluations
N_EVAL_EPISODES = 10   # episodes averaged per evaluation


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
    """Online Q-learning. Returns best mean_return (float)."""
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
    gamma            = float(config.get("gamma", 0.99))
    stimuli_t        = stimuli_to_tensor(rdm_inputs)
    checkpoint_steps = set(log4_checkpoints(max_steps))

    history = []

    env         = env_factory()
    global_step = 0
    obs, _      = env.reset()

    best_return  = float("-inf")
    best_step    = 0
    final_return = 0.0

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

        if global_step in checkpoint_steps:
            save_activations_mlp(model, stimuli_t,
                                 run_dir / f"step_{global_step:07d}", device)

        if global_step % EVAL_INTERVAL == 0:
            mean_ret = _eval_mean_return(model, env_factory, N_EVAL_EPISODES, device)
            history.append({"step": global_step, "mean_return": round(float(mean_ret), 4)})
            if mean_ret > best_return:
                best_return = mean_ret
                best_step   = global_step
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            if verbose:
                tag = "  *** SOLVED ***" if mean_ret >= task.success_threshold else ""
                print(f"  step {global_step:8,}  mean_return={mean_ret:7.2f}{tag}", flush=True)
            if mean_ret >= task.success_threshold:
                break

        obs = next_obs if not done else env.reset()[0]

    env.close()

    # Final eval if last eval wasn't at global_step
    if not history or history[-1]["step"] != global_step:
        mean_ret = _eval_mean_return(model, env_factory, N_EVAL_EPISODES, device)
        history.append({"step": global_step, "mean_return": round(float(mean_ret), 4)})
        if mean_ret > best_return:
            best_return = mean_ret
            best_step   = global_step
            torch.save(model.state_dict(), run_dir / "model_best.pt")

    final_return = history[-1]["mean_return"]

    # Save final activations from current (end-of-training) weights
    save_activations_mlp(model, stimuli_t, run_dir / "final", device)

    # Save best activations by reloading best weights
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    save_activations_mlp(model, stimuli_t, run_dir / "best", device)

    metadata = {
        "task":         task.name,
        "paradigm":     task.paradigm,
        "config":       dict(config),
        "best_step":    best_step,
        "best_metric":  round(float(best_return), 4),
        "final_step":   global_step,
        "final_metric": round(float(final_return), 4),
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return float(best_return)
