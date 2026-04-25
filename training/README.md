# Citadel — Training

See [docs/training.md](../docs/training.md) for the full training pipeline guide.

## Quick start (Colab T4)

```python
%cd /content
!rm -rf /content/citadel
!git clone https://github.com/Astro-Dude/citadel.git /content/citadel
%cd /content/citadel

import os
os.environ["PHASE"]     = "1"          # "1", "2", or "both"
os.environ["MAX_STEPS"] = "120"
os.environ["N_SEEDS"]   = "6"
os.environ["SAVE_DIR"]  = "/content/checkpoints"

!python training/grpo_train.py
```
