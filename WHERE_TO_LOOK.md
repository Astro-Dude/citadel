# Where to look (judge's-eye TL;DR)

Reading the whole [README](README.md) is rewarded but optional. If you have
**three minutes**, here is the shortest path through Citadel:

1. **[playbook_export.md](playbook_export.md)** — a deterministic baseline
   council run across all tasks × adversary generations, with the shared
   playbook rendered as Markdown. The `Top cited` table and `wins/losses`
   columns show the citation/utility mechanic working end-to-end (no LLM
   needed). Regenerate locally with `python scripts/demo_export.py`.

2. **[examples/single_episode.py](examples/single_episode.py)** — one
   episode of `easy_1`, three steps, prints the `propose → critique →
   execute` loop step-by-step, including a concrete `VETO` and the trust
   score moving in response. ~30 seconds to read.

3. **[README.md § Why a council, not a gate](README.md#why-a-council-not-a-gate)**
   — the structured-critique design, in 100 words, with the JSON schema
   that Oversight emits every step.

4. **[README.md § Scoring architecture](README.md#scoring-architecture)** —
   the full multi-layer reward formula (six sub-scores summing to 1.00,
   with the catastrophic-zeroing rule and severity ramp).

For deeper inspection, the `runs/` directory contains full transcripts
(`transcript.md` per task) when `python inference.py` is run with an LLM.
