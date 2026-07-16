"""One-off migration: rename all RDM dataset keys to add _cosine suffix.

Renames e.g. layer_0 → layer_0_cosine, temporal → temporal_cosine in every
checkpoint group across all task HDF5 files. Uses h5py Group.move() which is
an in-place HDF5 link rename — no data is copied.

Idempotent: keys already ending in _cosine or _pearson are skipped.
"""
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).parent))
from analysis_utils import RDM_DIR


def migrate_file(h5_path: Path) -> tuple[int, int]:
    n_renamed = 0
    n_skipped = 0

    with h5py.File(h5_path, "a") as h5:
        runs_grp = h5.get("runs")
        if runs_grp is None:
            return 0, 0

        for run_id in runs_grp.keys():
            rg = runs_grp[run_id]
            for ckpt_name in rg.keys():
                ckpt_grp = rg[ckpt_name]
                # Collect dataset keys to rename
                to_rename = []
                for key in list(ckpt_grp.keys()):
                    if not isinstance(ckpt_grp[key], h5py.Dataset):
                        continue
                    if key.endswith("_cosine") or key.endswith("_pearson"):
                        n_skipped += 1
                    else:
                        to_rename.append(key)
                for key in to_rename:
                    dest = f"{key}_cosine"
                    if dest in ckpt_grp:
                        raise RuntimeError(
                            f"Both '{key}' and '{dest}' exist in "
                            f"{run_id}/{ckpt_name} — resolve manually "
                            "before re-running"
                        )
                    ckpt_grp.move(key, dest)
                    n_renamed += 1

    return n_renamed, n_skipped


def main():
    h5_files = sorted(RDM_DIR.glob("*_rdms.h5"))
    if not h5_files:
        print(f"No HDF5 files found in {RDM_DIR}")
        return

    total_renamed = 0
    total_skipped = 0

    for h5_path in h5_files:
        task = h5_path.stem.replace("_rdms", "")
        print(f"  {task} ...", end=" ", flush=True)
        n_renamed, n_skipped = migrate_file(h5_path)
        print(f"{n_renamed} renamed, {n_skipped} already labelled")
        total_renamed += n_renamed
        total_skipped += n_skipped

    print(f"\nDone. {total_renamed} keys renamed, {total_skipped} already had metric suffix.")


if __name__ == "__main__":
    main()
