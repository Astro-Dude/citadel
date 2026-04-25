"""
Citadel — Before / After Evaluation
=====================================
Runs the same set of episodes with:
  (A) Untrained base model (Qwen2.5-3B-Instruct, greedy)
  (B) Trained Commander checkpoint

Produces a comparison table + per-metric bar chart that shows judges
measurable improvement across 6 axes.

Usage (Colab):
  python /content/Citadel/training/eval_before_after.py \\
      --trained_path /content/checkpoints/commander/final \\
      --n_episodes 12 \\
      --save_dir /content/checkpoints/eval

Output files:
  eval/before_after_table.md     — human-readable comparison table
  eval/before_after.json         — raw numbers
  eval/before_after_chart.png    — bar chart (6 metrics × 2 models)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Any

SCRIPT_DIR = Path(__file__).parent
CITADEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CITADEL_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from environment import CitadelEnvironment
from models import IncidentAction, OversightAction, OversightDecision
from baseline import oversight_rule_based
from inference import (
    COMMANDER_SYSTEM_PROMPT,
    format_commander_observation,
    parse_commander_response,
    ACTION_NAMES,
    SYSTEM_NAMES,
)

# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------

EVAL_TASKS = [
    ("easy_1",   1, list(range(4))),
    ("medium_1", 2, list(range(4))),
    ("hard_1",   3, list(range(4))),
]

METRICS = [
    "final_score",
    "bastion_v1_final_score",
    "governance_compliance",
    "oversight_precision",
    "investor_score",
    "data_exfiltrated",    # lower is better
]

MAX_STEPS = 12


# ---------------------------------------------------------------------------
# Run one episode with a given model
# ---------------------------------------------------------------------------

def run_episode(
    model,
    tokenizer,
    task_id: str,
    adversary_gen: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    env = CitadelEnvironment(oversight_policy=oversight_rule_based)
    obs = env.reset(task_id=task_id, seed=seed, adversary_gen=adversary_gen)
    obs_dict = obs.model_dump()

    history: List[str] = []
    final_meta: Dict[str, Any] = {}

    for step in range(MAX_STEPS):
        if obs.done:
            break

        user_msg = format_commander_observation(obs_dict, step=step, history=history)
        messages = [
            {"role": "system", "content": COMMANDER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1800
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=280,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        action = parse_commander_response(completion)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        meta = obs.metadata or {}

        action_name = ACTION_NAMES.get(action.action, str(action.action))
        tgt = SYSTEM_NAMES[action.target_system] if 0 <= action.target_system < len(SYSTEM_NAMES) else "?"
        history.append(f"Hour {step + 1}: {action_name}({tgt}) -> r={obs.reward:.2f}")

        if obs.done:
            final_meta = meta
            break

    if not final_meta:
        final_meta = obs.metadata or {}

    fs = final_meta.get("final_scores") or {}
    return {
        "task_id": task_id,
        "adversary_gen": adversary_gen,
        "seed": seed,
        "final_score":            float(fs.get("final_score", 0.0)),
        "bastion_v1_final_score": float(fs.get("bastion_v1_final_score", 0.0)),
        "governance_compliance":  float(fs.get("governance_compliance", 0.0)),
        "oversight_precision":    float(fs.get("oversight_precision", 0.0)),
        "investor_score":         float(fs.get("investor_score", 0.5)),
        "data_exfiltrated":       float(final_meta.get("data_exfiltrated", 0.5)),
        "steps": MAX_STEPS,
    }


def eval_model(
    model_name_or_path: str,
    label: str,
    n_episodes: int = 12,
) -> List[Dict[str, Any]]:
    print(f"\n[eval] Loading {label}: {model_name_or_path}")

    # Try Unsloth first (faster on T4), fall back to vanilla HF
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print(f"[eval] Loaded with Unsloth (4-bit)")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        model.eval()
        print(f"[eval] Loaded with vanilla HF")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    results = []
    total = sum(len(seeds) for _, _, seeds in EVAL_TASKS)
    done = 0

    for task_id, gen, seeds in EVAL_TASKS:
        for seed in seeds:
            done += 1
            print(f"[eval] {label} {done}/{total} — {task_id} gen{gen} seed{seed}", end="  ")
            try:
                r = run_episode(model, tokenizer, task_id, gen, seed, str(device))
                results.append(r)
                print(f"score={r['final_score']:.3f} exfil={r['data_exfiltrated']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "task_id": task_id, "adversary_gen": gen, "seed": seed,
                    "final_score": 0.0, "bastion_v1_final_score": 0.0,
                    "governance_compliance": 0.0, "oversight_precision": 0.5,
                    "investor_score": 0.5, "data_exfiltrated": 1.0, "steps": 0,
                })

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {m: 0.0 for m in METRICS}
    return {
        m: round(statistics.mean(r[m] for r in results), 4)
        for m in METRICS
    }


METRIC_LABELS = {
    "final_score":            "Final Score  (↑)",
    "bastion_v1_final_score": "Incident Outcome (↑)",
    "governance_compliance":  "Governance (↑)",
    "oversight_precision":    "Oversight Precision (↑)",
    "investor_score":         "Investor Satisfaction (↑)",
    "data_exfiltrated":       "Data Exfiltrated (↓)",
}

HIGHER_BETTER = {m: True for m in METRICS}
HIGHER_BETTER["data_exfiltrated"] = False


def format_delta(before: float, after: float, higher_better: bool) -> str:
    delta = after - before
    improved = delta > 0 if higher_better else delta < 0
    sign = "+" if delta > 0 else ""
    mark = "✓" if improved else "✗"
    return f"{sign}{delta:.4f} {mark}"


def write_markdown_table(
    before_agg: Dict[str, float],
    after_agg: Dict[str, float],
    save_path: Path,
    baseline_label: str,
    trained_label: str,
) -> None:
    lines = [
        f"# Citadel — Before vs After Training",
        f"",
        f"| Metric | {baseline_label} | {trained_label} | Delta |",
        f"|--------|{'---' * 4}|{'---' * 4}|-------|",
    ]
    for m in METRICS:
        label = METRIC_LABELS[m]
        b = before_agg[m]
        a = after_agg[m]
        delta_str = format_delta(b, a, HIGHER_BETTER[m])
        lines.append(f"| {label} | {b:.4f} | {a:.4f} | {delta_str} |")

    lines += [
        "",
        "## Per-task breakdown",
        "",
    ]
    save_path.write_text("\n".join(lines))
    print(f"[eval] table → {save_path}")


def write_chart(
    before_agg: Dict[str, float],
    after_agg: Dict[str, float],
    save_path: Path,
    baseline_label: str,
    trained_label: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # For data_exfiltrated, invert so "higher bar = better" on all axes
        display_before = []
        display_after = []
        labels = []
        for m in METRICS:
            b, a = before_agg[m], after_agg[m]
            if not HIGHER_BETTER[m]:
                b, a = 1.0 - b, 1.0 - a
            display_before.append(b)
            display_after.append(a)
            labels.append(METRIC_LABELS[m].split("(")[0].strip())

        x = np.arange(len(METRICS))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        bars_b = ax.bar(x - width / 2, display_before, width, label=baseline_label, color="#5b8dd9", alpha=0.85)
        bars_a = ax.bar(x + width / 2, display_after,  width, label=trained_label,  color="#2ecc71", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (higher = better on all axes)")
        ax.set_title("Citadel — Commander: Before vs After GRPO Training")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar in list(bars_b) + list(bars_a):
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] chart → {save_path}")
    except Exception as e:
        print(f"[eval] chart failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Citadel before/after eval")
    parser.add_argument("--base_model",    default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF model name or path for untrained baseline")
    parser.add_argument("--trained_path",  required=True,
                        help="Path to trained Commander checkpoint")
    parser.add_argument("--n_episodes",    type=int, default=12)
    parser.add_argument("--save_dir",      default="/content/checkpoints/eval")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run before
    before_results = eval_model(args.base_model, "Baseline (untrained)", args.n_episodes)
    before_agg = aggregate(before_results)

    # Run after
    after_results = eval_model(args.trained_path, "Trained (GRPO)", args.n_episodes)
    after_agg = aggregate(after_results)

    # Save raw
    raw = {"before": before_results, "after": after_results,
           "before_agg": before_agg, "after_agg": after_agg}
    (save_dir / "before_after.json").write_text(json.dumps(raw, indent=2))

    # Reports
    write_markdown_table(
        before_agg, after_agg,
        save_dir / "before_after_table.md",
        "Baseline (untrained)", "Trained (GRPO)",
    )
    write_chart(
        before_agg, after_agg,
        save_dir / "before_after_chart.png",
        "Baseline (untrained)", "Trained (GRPO)",
    )

    # Print summary
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<30} {'Before':>8} {'After':>8} {'Delta':>10}")
    print("-" * 55)
    for m in METRICS:
        b, a = before_agg[m], after_agg[m]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        better = (delta > 0) == HIGHER_BETTER[m]
        mark = "✓" if better else "✗"
        print(f"{METRIC_LABELS[m]:<30} {b:>8.4f} {a:>8.4f} {sign}{delta:>8.4f} {mark}")
    print("=" * 55)


if __name__ == "__main__":
    main()
