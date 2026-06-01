# Hyperparameters and Geometry

Follow-up to Devolder, Colin & Holroyd (submitted 2026), which used a grid search over hyperparameter conditions to study the relationship between hyperparameters and representational geometry in neural networks. This project extends the work primarily by adding 8 additional tasks (from 1 to 9). It also adds depth (1–2 hidden layers), replacing grid search with Bayesian optimisation (with a saturating acquisition function), and saving activations at logarithmically-spaced training checkpoints to study how representational geometry evolves during learning.

See [METHODS.md](METHODS.md) for full task and training details, and
[BO_DESIGN.md](BO_DESIGN.md) for acquisition function design rationale.

---

## Tasks

Nine tasks across three paradigms:

| Task | Key | Paradigm |
|---|---|---|
| MNIST dual-task | `mnist_dual` | Supervised MLP |
| MNIST 10-way | `mnist_10way` | Supervised MLP |
| Fashion-MNIST 10-way | `fashion_10way` | Supervised MLP |
| Spirals (3-arm) | `spirals` | Supervised MLP |
| 8-bit Parity | `parity` | Supervised MLP |
| MNIST row-by-row | `mnist_rnn` | RNN |
| Adding problem | `adding` | RNN |
| CartPole-v1 | `cartpole` | RL |
| FourRooms | `fourrooms` | RL |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running locally

**Single task:**
```bash
python run_bo.py --task spirals --n-iter 300 --h 0.15 --beta 4.0
```
Results go to `output/experiments/spirals/`.

**Multiple supervised tasks in parallel:**
```bash
python scripts/run_supervised_tests.py --parallel 2
```
Runs `mnist_dual`, `mnist_10way`, `fashion_10way`, `parity` — two at a time.
Results go to `output/experiments_supervised_test/`.

Interrupted runs resume automatically from the existing `bo_state.json`.

---

## Running on AWS

See [AWS_SETUP.md](AWS_SETUP.md). In brief:

1. Upload any existing `bo_state.json` to S3 (`tom-hyperparams-representations/<task>/`)
2. Launch an EC2 instance with the contents of `scripts/aws_startup_gp_test.sh`
   as User Data, changing `TASK_NAME` at the top
3. The instance clones the repo, installs dependencies, runs, uploads state to S3
   after every iteration, and self-terminates

---

## Output structure

```
output/
  experiments/          ← production BO runs (run_bo.py default)
    <task>/
      bo_state.json     ← full observation history
      run_NNNN_rR/      ← per-network output
        metadata.json
        history.json
        model_best.pt
        step_*.npz      ← activations at log₄-spaced training steps
        best.npz
        final.npz
  experiments_*/        ← test and exploratory runs
  figures/              ← plots
```

---

## Key files

| File | Purpose |
|---|---|
| `run_bo.py` | Main BO loop — Sobol phase then GP phase |
| `run_task.py` | Single-task runner (no BO) |
| `src/bo.py` | GP, acquisition function (UCBoverNeff), suggest_next |
| `src/train_supervised.py` | MLP training loop |
| `src/train_rnn.py` | RNN training loop |
| `src/train_rl.py` | Q-learning training loop |
| `tasks/` | One file per task; defines data, architecture, hyperparameter space |
| `scripts/` | Test runners, visualisation scripts, AWS startup scripts |
