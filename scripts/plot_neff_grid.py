"""
Large N_eff comparison grid.
Rows: Euclidean (h=0.01/0.05/0.1/0.2) then Gower (same h values).
Cols: n=50, 100, 150, 200 observations.
"""
import json, math, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))          # for plot_acquisition_slice_v2
sys.path.insert(0, str(Path(__file__).parent.parent))   # for tasks, src

from tasks import TASKS
from src.bo import (
    _cont_params_for_task, cat_params_for_task,
    get_primary_observations, _cont_to_unit_val, ORDINAL_PARAMS,
)
from plot_acquisition_slice_v2 import (
    build_obs_matrix, build_grid_matrix, compute_neff, ord_to_unit, make_grid_vals,
)

STATE   = "output/experiments_gp_test/spirals/bo_state.json"
FIXED   = {
    "learning_rate": 1e-3, "l1_reg": 1e-6, "l2_reg": 1e-6,
    "depth": 2, "activation": "relu", "optimizer": "adam", "init_scale": 0.1,
}
H_VALS      = [0.01, 0.05, 0.1, 0.2]
SNAPSHOTS   = [50, 100, 150, 200]
DIST_MODES  = [("Euclidean", False), ("Gower", True)]

Path("output/figures").mkdir(parents=True, exist_ok=True)
obs_all   = json.loads(Path(STATE).read_text())
primaries = get_primary_observations(obs_all)

task        = TASKS["spirals"]()
cont_params = _cont_params_for_task(task)
cat_params  = cat_params_for_task(task)

bs_vals, hs_vals = make_grid_vals(cont_params)
grid_mat = build_grid_matrix(bs_vals, hs_vals, FIXED, cont_params, cat_params)

snap_obs     = [primaries[:n] for n in SNAPSHOTS]
snap_obs_mat = [build_obs_matrix(obs, cont_params, cat_params) for obs in snap_obs]

n_rows = len(DIST_MODES) * len(H_VALS)   # 8
n_cols = len(SNAPSHOTS)                   # 4

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows))
fig.suptitle("N_eff — relu/adam/depth=2/init_scale=0.1 slice", fontsize=13, y=1.001)

for dist_idx, (dist_name, normalize) in enumerate(DIST_MODES):
    for h_idx, h in enumerate(H_VALS):
        row = dist_idx * len(H_VALS) + h_idx
        for col, (obs, obs_mat) in enumerate(zip(snap_obs, snap_obs_mat)):
            ax  = axes[row, col]
            nef = compute_neff(grid_mat, obs_mat, cont_params, cat_params,
                               h=h, normalize=normalize)
            mat = nef.reshape(len(bs_vals), len(hs_vals))
            im  = ax.pcolormesh(hs_vals, bs_vals, mat, cmap="plasma", shading="auto")
            plt.colorbar(im, ax=ax)

            # Red crosses for all networks (no transparency — just show positions)
            hs_obs = [o["config"]["hidden_size"] for o in obs]
            bs_obs = [o["config"]["batch_size"]  for o in obs]
            ax.scatter(hs_obs, bs_obs, marker="+", c="white", s=30,
                       linewidths=0.8, alpha=0.5, zorder=5)

            ax.set_xscale("log"); ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"n={SNAPSHOTS[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{dist_name}  h={h}\nbatch_size", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("hidden_size", fontsize=9)

    # Dividing line between distance blocks
    if dist_idx == 0:
        y_mid = (len(H_VALS) - 0.5) / n_rows
        fig.add_artist(plt.Line2D([0.02, 0.98], [1 - y_mid, 1 - y_mid],
                                  transform=fig.transFigure,
                                  color="gray", linewidth=1.2, linestyle="--"))

plt.tight_layout()
plt.savefig("output/figures/neff_grid.png", dpi=120, bbox_inches="tight")
print("Saved neff_grid.png")
