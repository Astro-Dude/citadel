# Citadel — Training Pipeline

Two-phase GRPO training on **Qwen2.5-3B-Instruct** using TRL + Unsloth.
Runs on a free Colab T4 (16 GB) in under 20 minutes per phase.

---

## Files

| File | Purpose |
|------|---------|
| `training/grpo_train.py` | Phase 1 (Commander) + Phase 2 (Oversight) GRPO training |
| `training/eval_before_after.py` | Runs untrained vs trained on 12 episodes, produces comparison table + chart |
| `training/curriculum_eval.ipynb` | Curriculum progression analysis |
| `training/trust_analysis.ipynb` | Trust dynamics evolution plots |
| `training/train_commander.ipynb` | Older notebook (superseded by grpo_train.py) |
| `training/train_oversight.ipynb` | Older notebook (superseded by grpo_train.py) |

---

## Platform Support

| Platform | Backend | Speed | Notes |
|---|---|---|---|
| **Google Colab T4** | Unsloth 4-bit QLoRA | ~15 min/phase | Recommended — free, no setup |
| **Mac (Apple Silicon)** | PEFT + bf16 on MPS | ~2–4 hrs/phase | M1/M2/M3/M4, works out of the box |
| **Windows / Linux (NVIDIA)** | Unsloth 4-bit QLoRA | ~15–30 min/phase | CUDA 11.8+, ~8 GB VRAM |
| **CPU-only** | PEFT + fp32 | Very slow | Testing only |

---

## Google Colab (recommended)

### 1. Open a T4 GPU runtime
Runtime → Change runtime type → **T4 GPU**

### 2. Clone the repo
```python
%cd /content
!rm -rf /content/citadel
!git clone https://github.com/Astro-Dude/citadel.git /content/citadel
%cd /content/citadel
```

### 3. Run Phase 1 — Train Commander (~15 min)
```python
import os
os.environ["PHASE"]     = "1"
os.environ["MAX_STEPS"] = "120"
os.environ["N_SEEDS"]   = "6"
os.environ["SAVE_DIR"]  = "/content/checkpoints"

!python training/grpo_train.py
```

Deps install automatically — Unsloth + TRL 0.20.0 are pinned at runtime.

Outputs:
- `/content/checkpoints/commander/final/` — LoRA adapter
- `/content/checkpoints/commander/reward_curve.json`
- `/content/checkpoints/commander/reward_curve.png`

### 4. Run Phase 2 — Train Oversight (~15 min)
```python
os.environ["PHASE"] = "2"
!python training/grpo_train.py
```

Outputs:
- `/content/checkpoints/oversight/final/`
- `/content/checkpoints/oversight/reward_curve.{json,png}`

### 5. Run both phases at once
```python
os.environ["PHASE"] = "both"
!python training/grpo_train.py
```

### 6. Before/After evaluation
```python
!python training/eval_before_after.py \
    --trained_path /content/checkpoints/commander/final \
    --n_episodes 12 \
    --save_dir /content/checkpoints/eval
```

Outputs:
- `before_after_table.md` — markdown comparison table
- `before_after_chart.png` — bar chart (6 metrics × 2 models)
- `before_after.json` — raw episode data

### 7. Download results before session expires
```python
from google.colab import files
files.download('/content/checkpoints/commander/reward_curve.png')
files.download('/content/checkpoints/oversight/reward_curve.png')
files.download('/content/checkpoints/eval/before_after_chart.png')
files.download('/content/checkpoints/eval/before_after_table.md')
```

---

## Mac — Apple Silicon (M1/M2/M3/M4)

No GPU required — the script uses PyTorch MPS (Metal Performance Shaders) automatically.

```bash
git clone https://github.com/Astro-Dude/citadel.git && cd citadel

# Install deps (no bitsandbytes needed on MPS)
pip install torch trl peft transformers accelerate datasets matplotlib openai
```

```bash
# Phase 1 — Commander
PHASE=1 MAX_STEPS=120 N_SEEDS=6 SAVE_DIR=./checkpoints python training/grpo_train.py

# Phase 2 — Oversight
PHASE=2 MAX_STEPS=120 N_SEEDS=6 SAVE_DIR=./checkpoints python training/grpo_train.py
```

- Uses PEFT LoRA + bf16 (Unsloth is CUDA-only and is skipped automatically)
- Training is slower than a T4; reduce `MAX_STEPS=40` for faster iteration during dev
- M3 Max / M4 Pro with 36 GB+ unified memory can handle this comfortably

---

## Windows / Linux — NVIDIA GPU

Requires CUDA 11.8+ and ~8 GB VRAM (4-bit QLoRA via Unsloth).

```bash
git clone https://github.com/Astro-Dude/citadel.git && cd citadel

# Install PyTorch with CUDA (adjust cu121 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
pip install trl peft transformers accelerate datasets matplotlib openai bitsandbytes
```

**Windows (PowerShell):**
```powershell
$env:PHASE="both"; $env:MAX_STEPS="120"; $env:N_SEEDS="6"; $env:SAVE_DIR="./checkpoints"
python training/grpo_train.py
```

**Linux (bash):**
```bash
PHASE=both MAX_STEPS=120 N_SEEDS=6 SAVE_DIR=./checkpoints python training/grpo_train.py
```

Unsloth installs itself automatically when CUDA is detected. On Linux, WSL2 also works with the Linux instructions above.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PHASE` | `both` | `1` (Commander only), `2` (Oversight only), `both` |
| `MAX_STEPS` | `120` | GRPO steps per phase |
| `N_SEEDS` | `6` | Seeds per task/gen combo for dataset |
| `SAVE_DIR` | `/content/checkpoints` | Where checkpoints are saved |

---

## Backend Auto-Detection

The script auto-detects hardware and adjusts accordingly:

| Backend | How loaded | When |
|---------|-----------|------|
| CUDA (T4, RTX, etc.) | Unsloth 4-bit QLoRA | `torch.cuda.is_available()` |
| Apple Silicon (MPS) | PEFT + bf16 | `torch.backends.mps.is_available()` |
| CPU | PEFT + fp32 | Fallback (slow, testing only) |

---

## Reward Design (two independent functions — anti-hacking)

**Outcome reward (75% weight)**
- Runs one env step with the Commander's parsed action
- Returns `commander_step_reward` from the environment (containment + exfil + governance + trust)
- Clipped to `[-1, 1]` for gradient stability

**Format reward (25% weight)**
- Checks completion parses as valid JSON with `{action, target, justification}`
- Bonus for `method`, `rollback_plan`, `cited_lessons` — encourages rich output
- Returns `[-0.2, 0.35]`

Using two independent reward functions (per hackathon guide §8) reduces the risk of a single signal being hacked.

---

## Curriculum Schedule

| Steps | Tasks active | Adversary gens |
|-------|-------------|----------------|
| 0–40 | `easy_1` only | Gen 1 |
| 40–80 | `easy_1` + `medium_1` | Gen 1, 2 |
| 80+ | All three tasks | Gen 1, 2, 3 |

Starting with easy tasks ensures the model gets non-zero reward early, which is critical for GRPO to work.

---

## GRPO Config

```python
GRPOConfig(
    num_generations=4,           # 4 rollouts per prompt, ranked by reward
    learning_rate=5e-6,          # conservative for RL stability
    max_completion_length=300,   # enough for full JSON + justification
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    temperature=0.7,
)
```

LoRA: r=16, targeting all attention + MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

---

## Expected Improvement (after 120 steps on T4)

| Metric | Untrained | Trained (expected) |
|--------|-----------|-------------------|
| Final score | ~0.45 | ~0.62 |
| Governance compliance | ~0.10 | ~0.45 |
| Data exfiltrated | ~0.55 | ~0.30 |
| Oversight first-pass approve rate | ~50% | ~70% |

---

## Saving the Model Correctly

The training script saves LoRA adapters via `trainer.save_model()`.

**Do not merge to 16-bit naively** (per hackathon guide §16). To load in inference:

```python
# Using PEFT directly
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", ...)
model = PeftModel.from_pretrained(base, "/content/checkpoints/commander/final")
```

```python
# Using Unsloth (recommended for inference speed)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "/content/checkpoints/commander/final", load_in_4bit=True
)
FastLanguageModel.for_inference(model)
```

---

## Three-Phase Training Plan

| Phase | What trains | Steps | Notes |
|-------|------------|-------|-------|
| 1 | Commander | 120 | Curriculum easy → medium → hard |
| 2 | Oversight (frozen Commander) | 120 | Learns approve/veto/critique against trained Commander |
| 3 (optional) | Both jointly | 50 | Stabilizes trust dynamics |

Phase 2 uses the rule-based Commander baseline as the proposal source during dataset construction, then trains Oversight to critique against the trained Phase 1 Commander during rollouts.

---

## Viewing Results on the Dashboard

The dashboard is a self-contained HTML file that replays any run step-by-step. It needs a `dashboard.json` produced by running inference.

### Step 1 — Run inference to generate a transcript

**From Colab** (after training, while the session is alive):
```python
import os
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
os.environ["MODEL_NAME"]   = "Qwen/Qwen2.5-72B-Instruct"
os.environ["HF_TOKEN"]     = "hf_xxx"

!python inference.py
```

**Locally:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx

python inference.py
```

This runs the council over all tasks and writes `runs/<run_id>/dashboard.json` plus `transcript.json` and `transcript.md`.

### Step 2 — Download run output from Colab (if applicable)

```python
from google.colab import files
import os

os.system("zip -r /content/run_output.zip /content/citadel/runs/")
files.download("/content/run_output.zip")
```

Also download the reward curves before the session expires:
```python
files.download("/content/checkpoints/commander/commander_reward_curve.png")
files.download("/content/checkpoints/commander/commander_reward_curve.json")
```

### Step 3 — Drop the run folder into the repo (if downloaded from Colab)

```bash
unzip ~/Downloads/run_output.zip -d /path/to/citadel/
```

### Step 4 — Regenerate dashboard and open it

```bash
python dashboard.py        # re-embeds all runs/ into runs/dashboard.html
```

**macOS:**
```bash
open runs/dashboard.html
```

**Windows:**
```powershell
start runs/dashboard.html
```

**Linux:**
```bash
xdg-open runs/dashboard.html
```

The dashboard scans the `runs/` directory automatically — every subfolder with a `dashboard.json` appears as a selectable run. No config needed.

### Alternative — load a single run without regenerating

Open `runs/dashboard.html` directly in any browser, then click **LOAD JSON** in the top bar and pick any `runs/<run_id>/dashboard.json` from your filesystem. This works without running `dashboard.py` at all.
