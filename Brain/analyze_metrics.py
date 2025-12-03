#!/usr/bin/env python3
"""
Utility to inspect per-episode metrics emitted by run_brain_client.
Summaries highlight policy/value losses, draw rates, and evaluation trends.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_metrics(path: Path) -> List[Dict]:
    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize(entries: List[Dict], window: int) -> Dict:
    if not entries:
        return {}
    window = max(1, window)
    recent = entries[-window:]

    def avg(field):
        vals = [e[field] for e in recent if field in e and isinstance(e[field], (int, float))]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "episodes": len(entries),
        "window": len(recent),
        "policy_loss": avg("policy_loss"),
        "value_loss": avg("value_loss"),
        "entropy_term": avg("entropy_term"),
        "mean_reward": avg("mean_reward"),
        "draw_rate": sum(1 for e in recent if e.get("outcome") == 0.0) / len(recent),
    }

    x_moves = sum(e.get("x_moves", 0) for e in recent)
    o_moves = sum(e.get("o_moves", 0) for e in recent)
    total_moves = x_moves + o_moves
    if total_moves:
        summary["x_move_ratio"] = x_moves / total_moves
        summary["o_move_ratio"] = o_moves / total_moves

    random_eval = next(
        (e for e in reversed(entries) if "random_eval_loss_rate" in e), None)
    if random_eval:
        summary["random_eval_episode"] = random_eval["episode"]
        summary["random_eval_loss_rate"] = random_eval["random_eval_loss_rate"]
        summary["random_eval_draw_rate"] = random_eval["random_eval_draw_rate"]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize Brain metrics log")
    parser.add_argument("--metrics-log", type=str, required=True,
                        help="Path to the metrics JSONL file produced by run_brain_client")
    parser.add_argument("--window", type=int, default=200,
                        help="Rolling window size for averages")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_log).expanduser()
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    entries = load_metrics(metrics_path)
    summary = summarize(entries, args.window)
    if not summary:
        print("No metrics found.")
        return

    print(f"Loaded {summary['episodes']} episodes from {metrics_path}")
    print(f"Rolling window: {summary['window']} episodes")
    print(f"Policy loss avg:  {summary.get('policy_loss')}")
    print(f"Value loss avg:   {summary.get('value_loss')}")
    print(f"Entropy term avg: {summary.get('entropy_term')}")
    print(f"Mean reward avg:  {summary.get('mean_reward')}")
    print(f"Draw rate:        {summary.get('draw_rate', 0.0)*100:.2f}%")
    if "x_move_ratio" in summary:
        print(f"Move balance X/O: {summary['x_move_ratio']:.3f} / {summary['o_move_ratio']:.3f}")
    if "random_eval_episode" in summary:
        print(f"Last random eval @ episode {summary['random_eval_episode']}: "
              f"loss={summary['random_eval_loss_rate']*100:.2f}% "
              f"draw={summary['random_eval_draw_rate']*100:.2f}%")


if __name__ == "__main__":
    main()

