"""Summarize hyperparameter sweep results from outputs/sweep/.

Reads per-run CSV logs and resolved configs to produce a sorted CSV results
table, including WandB run URLs where available.

Usage:
    python scripts/summarize_sweep.py [--sweep-dir DIR] [--logs-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/sweep"),
        help="Directory containing per-run sweep outputs (default: outputs/sweep)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing SLURM log files used to extract WandB URLs (default: logs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sweep_results.csv"),
        help="Output CSV file path (default: sweep_results.csv)",
    )
    return parser.parse_args()


def _read_final_auroc(run_dir: Path) -> float | None:
    csv_path = run_dir / "loggers" / "csv" / "version_0" / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        col = "final/val_auroc"
        if col not in df.columns:
            return None
        values = df[col].dropna()
        return float(values.iloc[-1]) if len(values) > 0 else None
    except Exception:
        return None


def _read_best_val_loss(run_dir: Path) -> float | None:
    csv_path = run_dir / "loggers" / "csv" / "version_0" / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        col = "val/loss"
        if col not in df.columns:
            return None
        values = df[col].dropna()
        return float(values.min()) if len(values) > 0 else None
    except Exception:
        return None


def _read_config_yaml(run_dir: Path) -> dict:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text())
        result = {}
        try:
            result["k"] = cfg["retriever"]["k"]
        except (KeyError, TypeError):
            result["k"] = ""
        try:
            result["lr"] = cfg["training"]["module"].get("lr", "")
        except (KeyError, TypeError, AttributeError):
            result["lr"] = ""
        try:
            result["enc_dim"] = cfg["encoder"]["embedding_dim"]
        except (KeyError, TypeError):
            result["enc_dim"] = ""
        try:
            result["epochs"] = cfg["training"]["trainer"]["max_epochs"]
        except (KeyError, TypeError):
            result["epochs"] = ""
        try:
            result["wandb_project"] = cfg.get("wandb_project", "")
        except (KeyError, TypeError, AttributeError):
            result["wandb_project"] = ""
        return result
    except Exception:
        return {}


def _find_wandb_run_id(run_dir: Path) -> str | None:
    """Extract WandB run ID from the local wandb directory name."""
    wandb_dir = run_dir / "loggers" / "wandb"
    if not wandb_dir.exists():
        return None
    for entry in wandb_dir.iterdir():
        # Directory names look like: run-20260407_010725-h3yp242y
        m = re.match(r"^run-\d{8}_\d{6}-(\w+)$", entry.name)
        if m:
            return m.group(1)
    return None


def _build_run_id_to_url(logs_dir: Path) -> dict[str, str]:
    """Scan SLURM log files for WandB 'View run at' lines and build run_id → URL map."""
    url_map: dict[str, str] = {}
    if not logs_dir.exists():
        return url_map
    # Strip ANSI escape codes before matching
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    url_pattern = re.compile(r"https://wandb\.ai/[^/]+/[^/]+/runs/(\w+)")
    for log_file in logs_dir.glob("*.out"):
        try:
            for line in log_file.read_text(errors="replace").splitlines():
                clean = ansi_escape.sub("", line)
                m = url_pattern.search(clean)
                if m and ("View run" in clean or "wandb.ai" in clean):
                    full_url = m.group(0)
                    run_id = m.group(1)
                    url_map[run_id] = full_url
        except Exception:
            continue
    # Also scan .err files
    for log_file in logs_dir.glob("*.err"):
        try:
            for line in log_file.read_text(errors="replace").splitlines():
                clean = ansi_escape.sub("", line)
                m = url_pattern.search(clean)
                if m:
                    full_url = m.group(0)
                    run_id = m.group(1)
                    url_map[run_id] = full_url
        except Exception:
            continue
    return url_map


def main() -> None:
    args = _parse_args()
    sweep_dir: Path = args.sweep_dir
    logs_dir: Path = args.logs_dir
    output_path: Path = args.output

    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found under {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    run_id_to_url = _build_run_id_to_url(logs_dir)

    rows = []
    for run_dir in run_dirs:
        auroc = _read_final_auroc(run_dir)
        val_loss = _read_best_val_loss(run_dir)
        cfg = _read_config_yaml(run_dir)
        run_id = _find_wandb_run_id(run_dir)
        wandb_url = run_id_to_url.get(run_id, "") if run_id else ""

        rows.append(
            {
                "name": run_dir.name,
                "k": cfg.get("k", ""),
                "lr": cfg.get("lr", ""),
                "enc_dim": cfg.get("enc_dim", ""),
                "epochs": cfg.get("epochs", ""),
                "val_auroc": auroc if auroc is not None else "",
                "best_val_loss": val_loss if val_loss is not None else "",
                "status": "done"
                if auroc is not None
                else ("running/failed" if val_loss is not None else "pending"),
                "wandb_url": wandb_url,
            }
        )

    # Sort by val_auroc descending (empty/missing last)
    rows.sort(key=lambda r: (r["val_auroc"] == "", -(r["val_auroc"] or 0)))

    fieldnames = ["name", "k", "lr", "enc_dim", "epochs", "val_auroc", "best_val_loss", "status", "wandb_url"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "done")
    print(f"Written {output_path}  ({done}/{len(rows)} runs complete)")


if __name__ == "__main__":
    main()
