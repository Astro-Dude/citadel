"""
Citadel — GRPO Training Script (Colab T4 + Apple Silicon ready)
===============================================================

Trains the Commander LLM via GRPO using TRL + Unsloth (CUDA) or
standard PEFT (Apple MPS / CPU).

Backend auto-detection:
  CUDA available  → Unsloth 4-bit QLoRA  (Colab T4, RTX, etc.)
  MPS available   → PEFT + bf16 on Apple Silicon (M1/M2/M3/M4)
  Neither         → PEFT + fp32 on CPU (slow, testing only)

Usage (Colab):
  1. Upload the Citadel/ folder to /content/Citadel  (or git clone)
  2. Run:  !python /content/Citadel/training/grpo_train.py

Usage (local Apple Silicon):
  pip install torch trl peft transformers accelerate datasets matplotlib
  SAVE_DIR=./checkpoints PHASE=1 python Citadel/training/grpo_train.py

Environment variables (optional):
  PHASE          1 (Commander only) | 2 (Oversight only) | both (default)
  MAX_STEPS      total GRPO steps per phase (default 120)
  N_SEEDS        seeds per task/gen combo for dataset (default 6)
  SAVE_DIR       checkpoint root (default /content/checkpoints)
"""

from __future__ import annotations

import json
import os
import sys
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Path setup — works whether run from Colab or locally
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
CITADEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CITADEL_ROOT))

SAVE_DIR = Path(os.getenv("SAVE_DIR", "/content/checkpoints"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = int(os.getenv("MAX_STEPS", "120"))
N_SEEDS = int(os.getenv("N_SEEDS", "6"))
PHASE = os.getenv("PHASE", "both")   # "1", "2", or "both"

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ---------------------------------------------------------------------------
# Backend detection — determines whether we use Unsloth (CUDA) or plain PEFT
# ---------------------------------------------------------------------------

import torch

def _detect_backend() -> str:
    """Returns 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

BACKEND = _detect_backend()
USE_UNSLOTH = (BACKEND == "cuda")
print(f"[backend] detected={BACKEND}  unsloth={'yes' if USE_UNSLOTH else 'no'}")

# ---------------------------------------------------------------------------
# Gemini investor client — OpenAI-compatible endpoint, falls back to None
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Investor LLM — set after model load in train_commander / train_oversight
# Uses the already-loaded Qwen model via QwenInvestorClient (no API key needed)
# ---------------------------------------------------------------------------
INVESTOR_LLM = None   # populated by _init_investor_llm() once model is loaded
GEMINI_MODEL  = ""    # unused — kept for import compat


# ---------------------------------------------------------------------------
# Install check
# ---------------------------------------------------------------------------

def _check_install():
    if USE_UNSLOTH:
        # Colab pre-installs TRL 0.23.x which has heavy optional deps (llm_blender,
        # mergekit) that break the import. Force-install a clean TRL 0.15.2 +
        # Unsloth so GRPOTrainer imports without those extras.
        print("[setup] CUDA detected — installing clean TRL + Unsloth stack...")
        os.system(
            "pip install -q --upgrade "
            "'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' "
            "datasets matplotlib accelerate openai"
        )
        # Pin TRL to 0.20.0: satisfies unsloth-zoo (>=0.18.2,!=0.19.0,<=0.24.0)
        # and predates the llm_blender/mergekit deps added in 0.23.x
        os.system("pip install -q 'trl==0.20.0'")
    else:
        missing = []
        for pkg in ["trl", "datasets", "peft", "transformers", "accelerate"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            print(f"[setup] Installing (non-CUDA): {', '.join(missing)}")
            os.system(
                "pip install -q "
                "trl peft transformers accelerate datasets matplotlib openai"
            )

_check_install()

# Reload trl after potential reinstall — importlib ensures we get the new version
import importlib, sys
for mod in list(sys.modules.keys()):
    if mod == "trl" or mod.startswith("trl."):
        sys.modules.pop(mod, None)

from datasets import Dataset
if USE_UNSLOTH:
    from unsloth import FastLanguageModel as _FLM  # noqa: ensure unsloth patches trl
from trl import GRPOConfig, GRPOTrainer

from environment import CitadelEnvironment
from models import IncidentAction, OversightAction, OversightDecision
from investor_agent import QwenInvestorClient
from inference import (
    COMMANDER_SYSTEM_PROMPT,
    OVERSIGHT_SYSTEM_PROMPT,
    format_commander_observation,
    format_oversight_observation,
    parse_commander_response,
    parse_oversight_response,
    ACTION_NAMES,
    SYSTEM_NAMES,
)
from baseline import oversight_rule_based

# ---------------------------------------------------------------------------
# Curriculum schedule
# Steps 0–40:   easy_1 only            (model learns basic investigate → isolate)
# Steps 40–80:  easy_1 + medium_1      (learns false positives, CAB prereqs)
# Steps 80+:    all three tasks         (deceptive APT, investor comms)
# ---------------------------------------------------------------------------

CURRICULUM = [
    (0,  40,  ["easy_1"],                    [1]),
    (40, 80,  ["easy_1", "medium_1"],        [1, 2]),
    (80, 9999, ["easy_1", "medium_1", "hard_1"], [1, 2, 3]),
]

def tasks_for_step(step: int):
    for start, end, tasks, gens in CURRICULUM:
        if start <= step < end:
            return tasks, gens
    return ["easy_1", "medium_1", "hard_1"], [1, 2, 3]


# ---------------------------------------------------------------------------
# Reward function 1: Outcome reward
# Run the Commander's parsed action in a fresh env reset to the same state.
# We use a single step from reset (step 0) — the prompt IS the reset obs.
# This is correct because each prompt in the dataset is a step-0 observation.
# ---------------------------------------------------------------------------

_outcome_parse_failures = 0
_outcome_calls = 0

def _outcome_reward(task_id: str, adversary_gen: int, seed: int, completion: str) -> float:
    """Run one env step with the parsed action, return commander step reward."""
    global _outcome_parse_failures, _outcome_calls
    _outcome_calls += 1
    try:
        action = parse_commander_response(completion)
    except Exception as e:
        _outcome_parse_failures += 1
        if _outcome_calls % 20 == 0:
            pct = 100 * _outcome_parse_failures / _outcome_calls
            print(f"[reward] parse failures: {_outcome_parse_failures}/{_outcome_calls} ({pct:.0f}%)")
        return -0.5
    try:
        env = CitadelEnvironment(
            oversight_policy=oversight_rule_based,
            investor_llm_client=INVESTOR_LLM,
            investor_model_name=GEMINI_MODEL if INVESTOR_LLM else "",
        )
        env.reset(task_id=task_id, seed=seed, adversary_gen=adversary_gen)
        obs = env.step(action)
        r = float(obs.reward or 0.0)
        return max(-1.0, min(1.0, r))
    except Exception as e:
        import traceback
        print(f"[reward] env crash: {e}\n{traceback.format_exc()[:600]}")
        return -0.3


# ---------------------------------------------------------------------------
# Reward function 2: Format reward (anti-hacking — independent of outcome)
# The completion must be parseable JSON with required fields.
# ---------------------------------------------------------------------------

def _format_reward(completion: str) -> float:
    import re, json as _json
    m = re.search(r"\{[\s\S]*\}", completion)
    if not m:
        return -0.2
    try:
        data = _json.loads(m.group())
    except Exception:
        return -0.1
    required = {"action", "target", "justification"}
    if not required.issubset(data.keys()):
        return 0.0
    # Bonus for richer output
    bonus = 0.0
    if data.get("method"):
        bonus += 0.05
    if data.get("rollback_plan"):
        bonus += 0.05
    if isinstance(data.get("cited_lessons"), list) and data["cited_lessons"]:
        bonus += 0.05
    return 0.15 + bonus


# ---------------------------------------------------------------------------
# Combined reward function (what GRPOTrainer calls)
# TRL passes: prompts (list[str]), completions (list[str]), **metadata columns
# ---------------------------------------------------------------------------

_reward_call_count = 0

def commander_reward_fn(
    prompts: List[str],
    completions: List[str],
    task_id: List[str] = None,
    adversary_gen: List[int] = None,
    seed: List[int] = None,
    **kwargs,
) -> List[float]:
    global _reward_call_count
    _reward_call_count += 1
    rewards = []
    outcomes, formats = [], []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        tid = (task_id[i] if task_id else "easy_1")
        gen = int(adversary_gen[i] if adversary_gen else 1)
        s = int(seed[i] if seed else 0)

        r_outcome = _outcome_reward(tid, gen, s, completion)
        r_format = _format_reward(completion)
        outcomes.append(r_outcome)
        formats.append(r_format)

        total = 0.75 * r_outcome + 0.25 * r_format
        rewards.append(float(total))

    # Log every 5 calls so we can see what the model is actually producing
    if _reward_call_count % 5 == 1:
        import statistics
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        print(f"[reward_fn] call={_reward_call_count} n={len(rewards)} "
              f"outcomes={[round(x,3) for x in outcomes]} "
              f"formats={[round(x,3) for x in formats]} "
              f"total_std={std:.4f}")
    return rewards


# ---------------------------------------------------------------------------
# Dataset builder — one prompt per (task, gen, seed) combination
# Each prompt is the step-0 Commander observation from a fresh env reset.
# ---------------------------------------------------------------------------

def build_commander_dataset(tokenizer, n_seeds: int = N_SEEDS) -> Dataset:
    examples = []
    tasks_all = ["easy_1", "medium_1", "hard_1"]
    gens_all = [1, 2, 3]

    for task_id in tasks_all:
        for gen in gens_all:
            for seed in range(n_seeds):
                try:
                    env = CitadelEnvironment(
            oversight_policy=oversight_rule_based,
            investor_llm_client=INVESTOR_LLM,
            investor_model_name=GEMINI_MODEL if INVESTOR_LLM else "",
        )
                    obs = env.reset(task_id=task_id, seed=seed, adversary_gen=gen)
                    obs_dict = obs.model_dump()
                    user_msg = format_commander_observation(obs_dict, step=0, history=[])
                    messages = [
                        {"role": "system", "content": COMMANDER_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    examples.append({
                        "prompt": prompt,
                        "task_id": task_id,
                        "adversary_gen": gen,
                        "seed": seed,
                    })
                except Exception as e:
                    print(f"[dataset] skip {task_id}/gen{gen}/seed{seed}: {e}")

    print(f"[dataset] built {len(examples)} Commander prompts")
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Load model — Unsloth on CUDA, standard PEFT on MPS/CPU
# ---------------------------------------------------------------------------

def load_model(model_name: str, max_seq_len: int = 2048):
    """Universal model loader. Uses Unsloth on CUDA, plain PEFT on MPS/CPU."""
    if USE_UNSLOTH:
        return _load_unsloth(model_name, max_seq_len)
    else:
        return _load_peft(model_name, max_seq_len)


def _load_unsloth(model_name: str, max_seq_len: int):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[model] loaded via Unsloth 4-bit QLoRA on CUDA")
    return model, tokenizer


def _load_peft(model_name: str, max_seq_len: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    dtype = torch.bfloat16 if BACKEND == "mps" else torch.float32
    device_map = {"": BACKEND}

    print(f"[model] loading {model_name} in {dtype} on {BACKEND} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    print(f"[model] loaded via PEFT LoRA on {BACKEND}")
    return model, tokenizer


# keep alias for any external references
load_model_unsloth = load_model


# ---------------------------------------------------------------------------
# Reward curve save + plot
# ---------------------------------------------------------------------------

def save_reward_curve(log_history: list, save_dir: Path, tag: str):
    rewards = [x.get("reward") for x in log_history if x.get("reward") is not None]
    curve_path = save_dir / f"{tag}_reward_curve.json"
    curve_path.write_text(json.dumps(rewards, indent=2))
    print(f"[curve] saved {len(rewards)} points → {curve_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rewards, linewidth=1.5)
        # Smooth rolling average
        if len(rewards) > 8:
            import statistics
            window = max(5, len(rewards) // 20)
            smoothed = [
                statistics.mean(rewards[max(0, i - window):i + 1])
                for i in range(len(rewards))
            ]
            ax.plot(smoothed, linewidth=2.5, color="C1", label=f"rolling avg (w={window})")
            ax.legend()
        ax.set_title(f"Citadel {tag} — Reward over GRPO steps")
        ax.set_xlabel("logging step")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.3)
        png_path = save_dir / f"{tag}_reward_curve.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[curve] plot saved → {png_path}")
    except Exception as e:
        print(f"[curve] matplotlib failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Investor LLM init — called once per phase after model is loaded
# ---------------------------------------------------------------------------

def _init_investor_llm(model, tokenizer):
    """Wire the loaded Qwen model into InvestorAgent as its LLM."""
    global INVESTOR_LLM
    try:
        INVESTOR_LLM = QwenInvestorClient(model, tokenizer, max_new_tokens=200)
        print("[investor] QwenInvestorClient ready — dynamic investor messages enabled")
    except Exception as e:
        print(f"[investor] QwenInvestorClient failed ({e}) — using rule-based fallback")
        INVESTOR_LLM = None


# ---------------------------------------------------------------------------
# Phase 1: Train Commander
# ---------------------------------------------------------------------------

def _grpo_dtype_flags() -> dict:
    """Returns the correct bf16/fp16 flags for the current backend."""
    if BACKEND == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        return {"bf16": bf16, "fp16": not bf16}
    if BACKEND == "mps":
        # MPS supports bfloat16 on M-series; skip fp16 (unstable on MPS)
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": False}


def train_commander():
    print("\n" + "=" * 60)
    print(f"PHASE 1 — Train Commander (GRPO, backend={BACKEND})")
    print("=" * 60)

    model, tokenizer = load_model(BASE_MODEL)
    _init_investor_llm(model, tokenizer)
    dataset = build_commander_dataset(tokenizer, n_seeds=N_SEEDS)

    cmd_save = SAVE_DIR / "commander"
    cmd_save.mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(
        output_dir=str(cmd_save),
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=6,          # balance: enough diversity, not too slow on T4
        max_completion_length=300,  # enough for JSON + justification
        temperature=1.1,            # slightly above 1.0 for output diversity
        logging_steps=5,
        save_steps=max(1, MAX_STEPS // 4),
        report_to="none",
        remove_unused_columns=False,
        **_grpo_dtype_flags(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[commander_reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[train] Starting Commander GRPO — {MAX_STEPS} steps, curriculum: easy→medium→hard")
    trainer.train()

    # Save final model
    final_path = cmd_save / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[train] Commander saved → {final_path}")

    save_reward_curve(trainer.state.log_history, cmd_save, "commander")
    return str(final_path)


# ---------------------------------------------------------------------------
# Phase 2: Train Oversight (frozen Commander)
# ---------------------------------------------------------------------------

def _oversight_outcome_reward(task_id: str, gen: int, seed: int, completion: str) -> float:
    """
    Run a full council step with:
      - Commander = rule-based baseline (frozen; Phase 2 focuses on Oversight)
      - Oversight  = parsed from the completion
    Returns the oversight_reward from env metadata.
    """
    try:
        from baseline import naive_policy
        env = CitadelEnvironment(
            oversight_policy=oversight_rule_based,
            investor_llm_client=INVESTOR_LLM,
            investor_model_name=GEMINI_MODEL if INVESTOR_LLM else "",
        )
        env.reset(task_id=task_id, seed=seed, adversary_gen=gen)

        # Get a naive Commander action
        action_idx, target_idx = naive_policy(env._state, env._state.hour)
        proposal_action = IncidentAction(
            action=action_idx, target_system=target_idx,
            justification="naive baseline action",
        )

        ov_action = parse_oversight_response(completion)
        obs = env.step(proposal_action, oversight_action=ov_action)
        r = float((obs.metadata or {}).get("oversight_reward", 0.0))
        return max(-1.0, min(1.0, r))
    except Exception:
        return -0.3


def _oversight_format_reward(completion: str) -> float:
    import re, json as _json
    m = re.search(r"\{[\s\S]*\}", completion)
    if not m:
        return -0.2
    try:
        data = _json.loads(m.group())
    except Exception:
        return -0.1
    required = {"decision", "risk_tier", "weakness", "lesson_text"}
    if not required.issubset(data.keys()):
        return 0.0
    bonus = 0.0
    if isinstance(data.get("missing_evidence"), list) and data["missing_evidence"]:
        bonus += 0.05
    if data.get("counter_proposal") and isinstance(data["counter_proposal"], dict):
        bonus += 0.05
    if len(str(data.get("lesson_text", ""))) > 20:
        bonus += 0.05
    return 0.15 + bonus


def oversight_reward_fn(
    prompts: List[str],
    completions: List[str],
    task_id: List[str] = None,
    adversary_gen: List[int] = None,
    seed: List[int] = None,
    **kwargs,
) -> List[float]:
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        tid = (task_id[i] if task_id else "easy_1")
        gen = int(adversary_gen[i] if adversary_gen else 1)
        s = int(seed[i] if seed else 0)
        r_outcome = _oversight_outcome_reward(tid, gen, s, completion)
        r_format = _oversight_format_reward(completion)
        total = 0.70 * r_outcome + 0.30 * r_format
        rewards.append(float(total))
    return rewards


def build_oversight_dataset(tokenizer, n_seeds: int = N_SEEDS) -> Dataset:
    """
    Build Oversight training prompts.
    Each prompt is the Oversight observation for a step-0 council turn,
    where the Commander proposal comes from the rule-based naive baseline.
    """
    from baseline import naive_policy

    examples = []
    for task_id in ["easy_1", "medium_1", "hard_1"]:
        for gen in [1, 2, 3]:
            for seed in range(n_seeds):
                try:
                    env = CitadelEnvironment(
            oversight_policy=oversight_rule_based,
            investor_llm_client=INVESTOR_LLM,
            investor_model_name=GEMINI_MODEL if INVESTOR_LLM else "",
        )
                    env.reset(task_id=task_id, seed=seed, adversary_gen=gen)

                    # Get naive Commander proposal
                    action_idx, target_idx = naive_policy(env._state, env._state.hour)
                    proposal_action = IncidentAction(
                        action=action_idx, target_system=target_idx,
                        justification="naive baseline action",
                    )

                    from models import CommanderProposal, ACTION_NAMES, SYSTEM_NAMES
                    oobs = {
                        "proposed_action": {
                            "action": proposal_action.action,
                            "action_name": ACTION_NAMES.get(proposal_action.action, "?"),
                            "target_system": proposal_action.target_system,
                            "target_name": SYSTEM_NAMES[proposal_action.target_system],
                            "method": proposal_action.method,
                            "scope": proposal_action.scope,
                            "rollback_plan": proposal_action.rollback_plan,
                            "severity_arg": proposal_action.severity_arg,
                            "channel_arg": proposal_action.channel_arg,
                            "message_arg": proposal_action.message_arg,
                            "scope_arg": proposal_action.scope_arg,
                            "evidence_arg": proposal_action.evidence_arg,
                        },
                        "justification": proposal_action.justification,
                        "cited_lessons": list(proposal_action.cited_lessons),
                        "policy_checks": {},
                        "veto_budget_remaining": 4,
                        "flag_budget_remaining": 2,
                        "shared_playbook": [],
                        "trust_summary": {},
                        "oversight_episode_history": [],
                        "raw_alert_digest": [
                            a.model_dump() for a in env._state.alerts[-4:]
                        ],
                    }
                    user_msg = format_oversight_observation(oobs)
                    messages = [
                        {"role": "system", "content": OVERSIGHT_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    examples.append({
                        "prompt": prompt,
                        "task_id": task_id,
                        "adversary_gen": gen,
                        "seed": seed,
                    })
                except Exception as e:
                    print(f"[dataset] skip oversight {task_id}/gen{gen}/seed{seed}: {e}")

    print(f"[dataset] built {len(examples)} Oversight prompts")
    return Dataset.from_list(examples)


def train_oversight(commander_path: str = None):
    print("\n" + "=" * 60)
    print(f"PHASE 2 — Train Oversight (GRPO, backend={BACKEND}, frozen Commander)")
    print("=" * 60)

    model, tokenizer = load_model(BASE_MODEL)
    _init_investor_llm(model, tokenizer)
    dataset = build_oversight_dataset(tokenizer, n_seeds=N_SEEDS)

    ov_save = SAVE_DIR / "oversight"
    ov_save.mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(
        output_dir=str(ov_save),
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=6,
        max_completion_length=300,
        temperature=1.1,
        logging_steps=5,
        save_steps=max(1, MAX_STEPS // 4),
        report_to="none",
        remove_unused_columns=False,
        **_grpo_dtype_flags(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[oversight_reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[train] Starting Oversight GRPO — {MAX_STEPS} steps")
    trainer.train()

    final_path = ov_save / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"[train] Oversight saved → {final_path}")

    save_reward_curve(trainer.state.log_history, ov_save, "oversight")
    return str(final_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(textwrap.dedent(f"""
    Citadel GRPO Training
    =====================
    Base model : {BASE_MODEL}
    Max steps  : {MAX_STEPS} per phase
    Dataset    : {N_SEEDS} seeds × 3 tasks × 3 gens = {N_SEEDS * 9} prompts
    Save dir   : {SAVE_DIR}
    Phase      : {PHASE}
    Curriculum : easy(0-40) → +medium(40-80) → +hard(80+)
    """))

    cmd_path = None
    if PHASE in ("1", "both"):
        cmd_path = train_commander()

    if PHASE in ("2", "both"):
        train_oversight(commander_path=cmd_path)

    print("\n[done] Training complete.")
    print(f"  Commander → {SAVE_DIR}/commander/final")
    print(f"  Oversight → {SAVE_DIR}/oversight/final")
    print(f"  Curves    → {SAVE_DIR}/*/reward_curve.{{json,png}}")
