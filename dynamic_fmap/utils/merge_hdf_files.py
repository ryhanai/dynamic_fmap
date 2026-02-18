"""
Merge ManiSkill demo HDF5 + JSON files under a directory structure like:

demos/
  PegInsertionSide-v1_1/teleop/trajectory.h5  trajectory.json
  PegInsertionSide-v1_2/teleop/trajectory.h5  trajectory.json
  PegInsertionSide-v1_3/teleop/trajectory.h5  trajectory.json

Outputs:
  demos/merged_teleop/trajectory.h5
  demos/merged_teleop/trajectory.json

Notes:
- Copies all HDF5 groups whose name starts with "traj_" and renumbers them sequentially.
- Merges JSON "episodes" if present, and renumbers episode_id / traj_id (if present).
- Keeps env_info / metadata from the first JSON, and warns if subsequent ones differ.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import tyro
from dataclasses import dataclass


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _json_deep_equal(a: Any, b: Any) -> bool:
    # Simple deep compare via canonical JSON; good enough for metadata checks.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _extract_traj_keys(h5: h5py.File) -> List[str]:
    keys = [k for k in h5.keys() if k.startswith("traj_")]
    # Sort by numeric suffix if present
    def keynum(k: str) -> Tuple[int, str]:
        m = re.match(r"traj_(\d+)$", k)
        return (int(m.group(1)) if m else 10**18, k)

    return sorted(keys, key=keynum)


def merge_h5(
    h5_paths: List[Path],
    out_h5_path: Path,
) -> int:
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    traj_counter = 0

    with h5py.File(out_h5_path, "w") as fout:
        for src_path in h5_paths:
            with h5py.File(src_path, "r") as fin:
                traj_keys = _extract_traj_keys(fin)
                if not traj_keys:
                    raise RuntimeError(f"No traj_* groups found in: {src_path}")

                for k in traj_keys:
                    new_k = f"traj_{traj_counter}"
                    fin.copy(k, fout, name=new_k)
                    traj_counter += 1

                # Optionally copy top-level attrs (only once)
                if traj_counter == len(traj_keys):
                    for attr_k, attr_v in fin.attrs.items():
                        fout.attrs[attr_k] = attr_v

    return traj_counter


def merge_json(
    json_paths: List[Path],
    out_json_path: Path,
    total_trajs: int,
) -> None:
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {}
    merged_episodes: List[Dict[str, Any]] = []
    base_env_info: Optional[Any] = None

    episode_counter = 0

    for idx, jp in enumerate(json_paths):
        data = _load_json(jp)

        # Keep top-level keys from first JSON (except episodes which we merge)
        if idx == 0:
            merged = {k: v for k, v in data.items() if k != "episodes"}
            base_env_info = data.get("env_info", None)
        else:
            # Warn if env_info differs
            if base_env_info is not None and "env_info" in data:
                if not _json_deep_equal(base_env_info, data["env_info"]):
                    print(f"[WARN] env_info differs in {jp} (keeping the first file's env_info).")

        eps = data.get("episodes", None)
        if eps is None:
            # Some formats may store episode metadata differently; we just skip.
            print(f"[WARN] No 'episodes' key in {jp}. JSON will only keep first file's metadata.")
            continue

        for ep in eps:
            ep2 = dict(ep)  # shallow copy
            # Renumber common fields if present
            if "episode_id" in ep2:
                ep2["episode_id"] = episode_counter
            if "traj_id" in ep2:
                ep2["traj_id"] = episode_counter
            if "traj_idx" in ep2:
                ep2["traj_idx"] = episode_counter

            merged_episodes.append(ep2)
            episode_counter += 1

    # Sanity check: number of episodes vs HDF5 trajectories
    if total_trajs != episode_counter:
        print(
            f"[WARN] HDF5 traj count ({total_trajs}) != JSON episode count ({episode_counter}). "
            "This can be OK if JSON lacks episodes or includes filtered episodes, but replay/index may mismatch."
        )

    merged["episodes"] = merged_episodes

    with out_json_path.open("w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def find_demo_files(root: Path) -> Tuple[List[Path], List[Path]]:
    h5s: List[Path] = []
    jss: List[Path] = []

    # Expect pattern: demos/*/teleop/trajectory.h5 and trajectory.json
    for teleop_dir in sorted(root.glob("*/teleop")):
        h5p = teleop_dir / "trajectory.h5"
        jsp = teleop_dir / "trajectory.json"
        if h5p.exists():
            h5s.append(h5p)
        else:
            print(f"[WARN] Missing {h5p}")
        if jsp.exists():
            jss.append(jsp)
        else:
            print(f"[WARN] Missing {jsp}")

    if not h5s:
        raise RuntimeError(f"No trajectory.h5 found under: {root}")

    # Keep paired ordering by directory name
    return h5s, jss


@dataclass
class Args:
    traj_root_dir: Path
    """directory that contains trajectory files to be merged"""
    out_dir: str = 'PegInsertionSide-v1_merged'
    """output directory to save merged trajectory files (HDF5 + JSON)"""
    method: str = 'teleop'


def merge_hdf_files(args: Args):
    root = Path(args.traj_root_dir)
    out_dir = root / args.out_dir / args.method
    out_h5 = out_dir / "trajectory.h5"
    out_json = out_dir / "trajectory.json"

    h5_paths, json_paths = find_demo_files(root)

    print("[INFO] Input HDF5 files:")
    for p in h5_paths:
        print("  -", p)

    print("[INFO] Input JSON files:")
    for p in json_paths:
        print("  -", p)

    total_trajs = merge_h5(h5_paths, out_h5)
    merge_json(json_paths, out_json, total_trajs)

    print(f"[DONE] Wrote:\n  {out_h5}\n  {out_json}\n  (merged traj count = {total_trajs})")


if __name__ == '__main__':
    args = tyro.cli(Args)
    merge_hdf_files(args)
