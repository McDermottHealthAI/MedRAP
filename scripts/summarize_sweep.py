"""Summarize hyperparameter sweep results from outputs/sweep/.

Reads per-run CSV logs and resolved configs to produce a sorted results table.

Usage:
    python scripts/summarize_sweep.py [--sweep-dir outputs/sweep]
"""

from __future__ import annotations

import argparse
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


def _read_config(run_dir: Path) -> dict:
    """Read key hyperparameters from resolved_config.yaml."""
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return {}
    try:
        import re

        text = config_path.read_text()

        def _find(pattern: str) -> str:
            m = re.search(pattern, text)
            return m.group(1).strip() if m else "?"

        return {
            "k": _find(r"^\s*k:\s*(\S+)", re.MULTILINE) if re.search(r"^\s*k:\s*(\S+)", text, re.MULTILINE) else "?",
            "lr": _find(r"^\s*lr:\s*(\S+)", re.MULTILINE) if re.search(r"^\s*lr:\s*(\S+)", text, re.MULTILINE) else "?",
            "enc_dim": _find(r"^\s*embedding_dim:\s*(\S+)", re.MULTILINE) if re.search(r"^\s*embedding_dim:\s*(\S+)", text, re.MULTILINE) else "?",
            "epochs": _find(r"^\s*max_epochs:\s*(\S+)", re.MULTILINE) if re.search(r"^\s*max_epochs:\s*(\S+)", text, re.MULTILINE) else "?",
        }
    except Exception:
        return {}


def _read_config_yaml(run_dir: Path) -> dict:
    """Read key hyperparameters using yaml if available, else regex fallback."""
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
            result["k"] = "?"
        try:
            result["lr"] = cfg["training"]["module"].get("lr", "?")
        except (KeyError, TypeError, AttributeError):
            result["lr"] = "?"
        try:
            result["enc_dim"] = cfg["encoder"]["embedding_dim"]
        except (KeyError, TypeError):
            result["enc_dim"] = "?"
        try:
            result["epochs"] = cfg["training"]["trainer"]["max_epochs"]
        except (KeyError, TypeError):
            result["epochs"] = "?"
        return result
    except Exception:
        return _read_config(run_dir)


def main() -> None:
    args = _parse_args()
    sweep_dir: Path = args.sweep_dir

    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found under {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        auroc = _read_final_auroc(run_dir)
        val_loss = _read_best_val_loss(run_dir)
        cfg = _read_config_yaml(run_dir)
        rows.append(
            {
                "name": run_dir.name,
                "k": cfg.get("k", "?"),
                "lr": cfg.get("lr", "?"),
                "enc_dim": cfg.get("enc_dim", "?"),
                "epochs": cfg.get("epochs", "?"),
                "val_auroc": auroc,
                "best_val_loss": val_loss,
                "status": "done" if auroc is not None else ("running/failed" if val_loss is not None else "pending"),
            }
        )

    # Sort by val_auroc descending (None last)
    rows.sort(key=lambda r: (r["val_auroc"] is None, -(r["val_auroc"] or 0)))

    header = f"{'name':<14} {'k':>4} {'lr':>8} {'enc_dim':>8} {'epochs':>6}  {'val_auroc':>10}  {'best_val_loss':>13}  {'status'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        auroc_str = f"{r['val_auroc']:.4f}" if r["val_auroc"] is not None else "     -"
        loss_str = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "            -"
        print(
            f"{r['name']:<14} {str(r['k']):>4} {str(r['lr']):>8} {str(r['enc_dim']):>8} {str(r['epochs']):>6}  {auroc_str:>10}  {loss_str:>13}  {r['status']}"
        )

    done = sum(1 for r in rows if r["val_auroc"] is not None)
    print(f"\n{done}/{len(rows)} runs complete.")


if __name__ == "__main__":
    main()
