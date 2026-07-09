# Sanity Check Analysis Plan

Goal: establish a basic picture of what we have before beginning the main analysis.
Coverage of hyperparameter space, where successful networks are concentrated,
and per-task performance distributions.

---

## Data facts (for reference)

**Paradigms and HP spaces**

| Paradigm   | Tasks                                          | Cont dims | Cat combos |
|------------|------------------------------------------------|-----------|------------|
| Supervised | mnist_dual, mnist_10way, fashion_10way, spirals, parity | 5 (lr, l1, l2, hidden_size, batch_size) | 24 (depth×activation×optimizer×init_scale) |
| RNN        | adding, mnist_rnn                              | 5 (same)  | 16 (cell_type×n_rnn_layers×optimizer×init_scale) |
| RL         | cartpole, fourrooms                            | 4 (lr, l1, l2, hidden_size; no batch_size) | 24 |

**Hypercube counts at b bins per continuous dim** (categorical dims treated as fixed levels)

| Bins/dim | Supervised | RNN    | RL    |
|----------|-----------|--------|-------|
| 2        | 768       | 512    | 384   |
| 3        | 5,832     | 3,888  | 1,944 |
| 4        | 24,576    | 16,384 | 6,144 |

With ~1000 networks per task, 2 bins/dim is the only option for joint coverage;
higher resolutions are used for marginal heatmaps only (where other dims are
collapsed).

**Activation files**: each run directory contains `best.npz` with key `layer_0`
of shape `(n_stimuli, hidden_size)`, and `metadata.json` with config +
best_metric. `bo_state.json` holds all observations with cont_unit_vals already
log-normalized to [0,1].

---

## Step 0 — Data loading module

**File**: `analysis/load_data.py`

Write reusable functions used by every subsequent script:

- `load_task_df(task_name, production_dir)` — loads `bo_state.json` and returns
  a flat DataFrame with one row per observation:
  - raw HP columns: `learning_rate`, `l1_reg`, `l2_reg`, `hidden_size`,
    `batch_size`, plus all categorical HPs
  - unit HP columns (log-normalized [0,1]): `unit_lr`, `unit_l1`, `unit_l2`,
    `unit_hidden`, `unit_batch` — pulled from `cont_unit_vals`
  - `performance`, `is_repeat`, `repeat_of`, `iteration`, `task`

- `load_all_tasks(production_dir)` — calls the above for all 9 tasks, returns
  one concatenated DataFrame, cached to `analysis/cache/all_observations.parquet`
  on first call (avoids re-parsing 9 JSON files on every run).

- `task_meta()` — returns a dict keyed by task name with: paradigm, chance_perf,
  success_threshold (None until Step 2), cont_param_names, cat_param_names,
  n_cat_combos.

---

## Step 1 — Disk inventory

**Script**: `analysis/01_disk_inventory.py`
**Output**: `analysis/tables/disk_inventory.csv`, printed summary

For each task, compare what is in `bo_state.json` vs. what is on disk:

- `n_in_state`: observations in bo_state.json
- `n_run_dirs`: run_NNNN_r0 directories present on disk
- `n_with_activations`: run dirs that contain `best.npz`
- `n_missing_activations`: run dirs without `best.npz` (crashed before save)
- `n_orphaned`: run dirs not referenced in bo_state.json (sanity check;
  should be 0 after the reconstruction script for mnist_dual)
- `n_primary`, `n_repeat`

This table becomes the ground truth for what data is available for all
subsequent analyses. Save it; it will be referenced again when loading
activations for the main analysis.

---

## Step 2 — Performance Lorenz curves and success thresholds

**Script**: `analysis/02_performance_lorenz.py`
**Output**: `analysis/figures/performance_lorenz.pdf`, `analysis/tables/success_thresholds.json`

For each task (primary observations only — exclude repeats):

1. Sort networks by performance ascending.
2. Plot cumulative performance (y) vs. network percentile (x), with the
   task's `chance_perf` drawn as a horizontal reference line.
3. Visually inspect where the curve lifts off the chance plateau.

After visual inspection, record a success threshold per task in
`analysis/tables/success_thresholds.json`. This file is read by all subsequent
steps that need a success/failure label.

Format:
```json
{
  "mnist_dual":    0.75,
  "mnist_10way":   0.40,
  ...
}
```

This step is **interactive**: run the script, view the figure, then fill in
`success_thresholds.json` manually before running Steps 3–6.

---

## Step 3 — Master summary table

**Script**: `analysis/03_summary_table.py`
**Output**: `analysis/tables/task_summary.csv`, printed

One row per task:

| Column             | Description |
|--------------------|-------------|
| task               | task name |
| paradigm           | supervised / rnn / rl |
| n_networks         | total observations |
| n_primary          | primary (non-repeat) |
| n_repeats          | repeat runs |
| repeat_rate        | n_repeats / n_networks |
| n_with_activations | from disk inventory |
| chance_perf        | from task definition |
| success_threshold  | from Step 2 |
| n_successful       | primary obs above threshold |
| pct_successful     | n_successful / n_primary |

This table is a useful header for the paper / supplement and a quick sanity
check. Print it to stdout as well for easy reading.

---

## Step 4 — Marginal coverage

**Script**: `analysis/04_marginal_coverage.py`
**Output**: `analysis/figures/marginal_coverage.pdf`, `analysis/tables/marginal_coverage.csv`

For each task, for each HP (using unit values for continuous):

**Continuous HPs** — divide [0,1] into 10 equal bins:
- fraction of bins with ≥1 primary network
- fraction of bins with ≥1 successful network

**Categorical HPs** — each level is its own bin:
- which levels have ≥1 primary network (should be all of them after 1000 runs)
- which levels have ≥1 successful network (may reveal dead zones)

Plot as a bar chart per task: one grouped bar per HP, two bars per HP
(total coverage vs. successful coverage). Lay out as a multi-panel figure
with one column per paradigm.

---

## Step 5 — Joint coverage (2-bin hypercubes)

**Script**: `analysis/05_joint_coverage.py`
**Output**: `analysis/tables/joint_coverage.csv`, printed

Assign each primary network to a hypercube using 2 bins per continuous dim
(bin 0: unit value < 0.5, bin 1: ≥ 0.5) × all categorical levels.

Per task, report:
- `n_hypercubes_total`: 768 / 512 / 384 depending on paradigm
- `n_occupied`: hypercubes with ≥1 network
- `pct_occupied`
- `n_with_success`: hypercubes with ≥1 successful network
- `pct_with_success`
- `mean_per_occupied`: mean networks per occupied cell
- `max_per_cell`: peak density

Append to `analysis/tables/task_summary.csv` or keep as a separate table —
either works since both will be read together.

---

## Step 6 — Concentration Lorenz curves and Gini coefficients

**Script**: `analysis/06_concentration_lorenz.py`
**Output**: `analysis/figures/concentration_lorenz.pdf`, `analysis/tables/gini.csv`

Using the same 2-bin hypercubes from Step 5, for each task:

1. Count successful networks per hypercube.
2. Sort hypercubes descending by successful-network count.
3. Lorenz curve: x = cumulative % of (occupied) hypercubes, y = cumulative %
   of all successful networks.
4. Gini coefficient = 2 × area between the curve and the diagonal.

Plot all tasks on one figure (one panel per paradigm, one line per task).
Add a reference diagonal. Print Gini per task.

Note: the saturation acquisition is designed to push Gini toward 0 (uniform
coverage). This plot is therefore also a check that the saturation mechanism
worked as intended.

---

## Step 7 — 2D marginal heatmaps

**Script**: `analysis/07_heatmaps.py`
**Output**: `analysis/figures/heatmaps_{task}.pdf` (one file per task)

For each task, for each pair of continuous HPs (10 pairs for supervised/RNN,
6 for RL — C(n_cont, 2)):

- Use 8 bins per axis (marginal: all other dims collapsed).
- Two side-by-side heatmaps per pair: (a) total network density, (b) success rate.
- Color scale: density uses sequential colormap; success rate uses diverging
  (centered at chance level for that task).

Lay out each task as a grid of pair panels. Since these are marginal (other dims
collapsed), 8 bins per axis is fine — no curse of dimensionality.

For categorical×continuous pairs (e.g., optimizer × learning_rate): replace
the categorical axis with one panel per categorical level rather than a heatmap.

---

## Output file summary

```
analysis/
  cache/
    all_observations.parquet       # cached flat DataFrame of all 9 tasks
  tables/
    disk_inventory.csv             # Step 1
    success_thresholds.json        # Step 2 (filled manually)
    task_summary.csv               # Steps 3 + 5
    marginal_coverage.csv          # Step 4
    joint_coverage.csv             # Step 5
    gini.csv                       # Step 6
  figures/
    performance_lorenz.pdf         # Step 2
    marginal_coverage.pdf          # Step 4
    concentration_lorenz.pdf       # Step 6
    heatmaps_{task}.pdf            # Step 7 (9 files)
  load_data.py                     # shared module, imported by all scripts
  01_disk_inventory.py
  02_performance_lorenz.py
  03_summary_table.py
  04_marginal_coverage.py
  05_joint_coverage.py
  06_concentration_lorenz.py
  07_heatmaps.py
```

---

## Execution order

Steps 0–2 must run before the rest. Step 2 requires manual inspection before
proceeding. Steps 3–7 can then run in any order (all read from the parquet cache
and `success_thresholds.json`).

```
python analysis/load_data.py          # generates cache (or auto-called by step 1)
python analysis/01_disk_inventory.py
python analysis/02_performance_lorenz.py
# → inspect figure, fill in analysis/tables/success_thresholds.json
python analysis/03_summary_table.py
python analysis/04_marginal_coverage.py
python analysis/05_joint_coverage.py
python analysis/06_concentration_lorenz.py
python analysis/07_heatmaps.py
```
