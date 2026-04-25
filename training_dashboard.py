"""
Citadel — Training Dashboard Generator

Scans checkpoints/<model-name>/ directories and produces a self-contained
runs/training_dashboard.html showing:
  - LLM selector (one entry per model found in checkpoints/)
  - Commander + Oversight reward curves (per step)
  - Per-checkpoint metrics table
  - Training config (LoRA rank, base model, etc.)
  - Comparison overlay (all models on same chart)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime

CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_PATH = Path("runs/training_dashboard.html")


def load_model_data(model_dir: Path) -> dict:
    """Load all training data for one model directory."""
    data = {
        "name": model_dir.name,
        "commander": None,
        "oversight": None,
    }

    for role in ("commander", "oversight"):
        role_dir = model_dir / role
        if not role_dir.exists():
            continue

        entry = {"steps": [], "rewards": [], "stds": [], "frac_zeros": [], "grad_norms": [], "config": {}}

        # Reward curve JSON (saved by grpo_train.py at end of training)
        curve_path = role_dir / f"{role}_reward_curve.json"
        if curve_path.exists():
            rewards = json.loads(curve_path.read_text())
            entry["rewards"] = rewards
            entry["steps"] = list(range(1, len(rewards) + 1))

        # Per-checkpoint trainer_state.json files for richer data
        checkpoint_dirs = sorted(role_dir.parent.glob(f"{role}/checkpoint-*") if role_dir.parent != role_dir else role_dir.glob("checkpoint-*"))
        # Also look in parent model dir
        checkpoint_dirs = sorted(model_dir.glob(f"{role}/checkpoint-*"))

        all_log_entries = []
        for ckpt in checkpoint_dirs:
            state_path = ckpt / "trainer_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                all_log_entries.extend(state.get("log_history", []))

        if all_log_entries:
            # Deduplicate by step, keep latest
            seen = {}
            for e in all_log_entries:
                seen[e["step"]] = e
            sorted_entries = sorted(seen.values(), key=lambda x: x["step"])
            entry["steps"] = [e["step"] for e in sorted_entries]
            entry["rewards"] = [e.get("reward", 0) for e in sorted_entries]
            entry["stds"] = [e.get("reward_std", 0) for e in sorted_entries]
            entry["frac_zeros"] = [e.get("frac_reward_zero_std", 0) for e in sorted_entries]
            entry["grad_norms"] = [min(e.get("grad_norm", 0), 50) for e in sorted_entries]  # cap at 50 for display

        # Adapter config for training metadata
        adapter_cfg = role_dir / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text())
            entry["config"] = {
                "base_model": cfg.get("base_model_name_or_path", "unknown"),
                "lora_r": cfg.get("r", "?"),
                "lora_alpha": cfg.get("lora_alpha", "?"),
                "target_modules": cfg.get("target_modules", []),
                "peft_version": cfg.get("peft_version", "?"),
            }

        data[role] = entry

    return data


def scan_checkpoints() -> list[dict]:
    """Scan checkpoints/ directory for model subdirectories."""
    if not CHECKPOINTS_DIR.exists():
        return []

    models = []
    for d in sorted(CHECKPOINTS_DIR.iterdir()):
        if d.is_dir() and (d / "commander").exists() or (d / "oversight").exists():
            models.append(load_model_data(d))

    return models


def build_html(models: list[dict]) -> str:
    models_json = json.dumps(models, indent=2)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Citadel — Training Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; background: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; }}
  .mono {{ font-family: 'JetBrains Mono', monospace; }}
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 2px; }}
  .tab-btn {{ transition: all 0.15s; }}
  .tab-btn.active {{ background: #1f6feb; color: #fff; }}
  .model-btn {{ transition: all 0.15s; border: 1px solid #30363d; }}
  .model-btn.active {{ border-color: #58a6ff; background: #1f2937; color: #58a6ff; }}
  .compare-btn.active {{ border-color: #3fb950; background: #1a2e1a; color: #3fb950; }}
  .metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
  .section-header {{ color: #8b949e; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }}
  canvas {{ max-height: 320px; }}
</style>
</head>
<body class="flex flex-col h-full">

<!-- Header -->
<div class="flex items-center justify-between px-6 py-3 border-b border-gray-800" style="background:#161b22">
  <div class="flex items-center gap-3">
    <span class="text-lg font-bold text-white">🏰 Citadel</span>
    <span class="text-gray-500 text-sm">Training Dashboard</span>
  </div>
  <div class="flex items-center gap-3">
    <button id="compareToggle" onclick="toggleCompare()"
      class="compare-btn px-3 py-1 rounded text-sm font-medium text-gray-400">
      Compare All Models
    </button>
    <span class="text-gray-600 text-xs mono">Generated {generated_at}</span>
  </div>
</div>

<!-- Model selector -->
<div class="px-6 py-3 border-b border-gray-800 flex items-center gap-2 flex-wrap" style="background:#0d1117" id="modelSelector">
  <span class="section-header mr-2">Model</span>
</div>

<!-- Tab bar -->
<div class="px-6 py-2 border-b border-gray-800 flex gap-1" style="background:#161b22">
  <button class="tab-btn active px-4 py-1.5 rounded text-sm font-medium" onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn px-4 py-1.5 rounded text-sm font-medium text-gray-400" onclick="switchTab('commander')">Commander</button>
  <button class="tab-btn px-4 py-1.5 rounded text-sm font-medium text-gray-400" onclick="switchTab('oversight')">Oversight</button>
  <button class="tab-btn px-4 py-1.5 rounded text-sm font-medium text-gray-400" onclick="switchTab('config')">Config</button>
</div>

<!-- Main content -->
<div class="flex-1 overflow-auto p-6" id="mainContent">

  <!-- Overview tab -->
  <div id="tab-overview">
    <div class="grid grid-cols-2 gap-4 mb-6" id="heroMetrics"></div>
    <div class="grid grid-cols-2 gap-4">
      <div class="metric-card p-4">
        <div class="section-header mb-3">Commander Reward</div>
        <canvas id="overviewCommanderChart"></canvas>
      </div>
      <div class="metric-card p-4">
        <div class="section-header mb-3">Oversight Reward</div>
        <canvas id="overviewOversightChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Commander tab -->
  <div id="tab-commander" class="hidden">
    <div class="grid grid-cols-3 gap-4 mb-4" id="cmdMetrics"></div>
    <div class="grid grid-cols-2 gap-4 mb-4">
      <div class="metric-card p-4">
        <div class="section-header mb-3">Reward per Step</div>
        <canvas id="cmdRewardChart"></canvas>
      </div>
      <div class="metric-card p-4">
        <div class="section-header mb-3">Gradient Norm</div>
        <canvas id="cmdGradChart"></canvas>
      </div>
    </div>
    <div class="grid grid-cols-2 gap-4">
      <div class="metric-card p-4">
        <div class="section-header mb-3">Reward Std Dev</div>
        <canvas id="cmdStdChart"></canvas>
      </div>
      <div class="metric-card p-4">
        <div class="section-header mb-3">Frac Zero Std (lower = better)</div>
        <canvas id="cmdFracChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Oversight tab -->
  <div id="tab-oversight" class="hidden">
    <div class="grid grid-cols-3 gap-4 mb-4" id="ovMetrics"></div>
    <div class="grid grid-cols-2 gap-4 mb-4">
      <div class="metric-card p-4">
        <div class="section-header mb-3">Reward per Step</div>
        <canvas id="ovRewardChart"></canvas>
      </div>
      <div class="metric-card p-4">
        <div class="section-header mb-3">Gradient Norm</div>
        <canvas id="ovGradChart"></canvas>
      </div>
    </div>
    <div class="grid grid-cols-2 gap-4">
      <div class="metric-card p-4">
        <div class="section-header mb-3">Reward Std Dev</div>
        <canvas id="ovStdChart"></canvas>
      </div>
      <div class="metric-card p-4">
        <div class="section-header mb-3">Frac Zero Std (lower = better)</div>
        <canvas id="ovFracChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Config tab -->
  <div id="tab-config" class="hidden">
    <div class="grid grid-cols-2 gap-4" id="configPanel"></div>
  </div>

</div>

<script>
const ALL_MODELS = {models_json};
const COLORS = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364'];

let selectedModel = ALL_MODELS.length > 0 ? ALL_MODELS[0].name : null;
let compareMode = false;
let activeTab = 'overview';
let charts = {{}};

function destroyChart(id) {{
  if (charts[id]) {{ charts[id].destroy(); delete charts[id]; }}
}}

function makeChart(id, labels, datasets, type='line', extra={{}}) {{
  destroyChart(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx, {{
    type,
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      animation: {{ duration: 300 }},
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
        tooltip: {{ backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
          titleColor: '#e6edf3', bodyColor: '#8b949e' }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
      }},
      ...extra
    }}
  }});
}}

function lineDs(label, data, color, fill=false) {{
  return {{
    label, data,
    borderColor: color,
    backgroundColor: fill ? color + '22' : 'transparent',
    borderWidth: 2,
    pointRadius: data.length > 30 ? 0 : 3,
    pointHoverRadius: 4,
    tension: 0.3,
    fill
  }};
}}

function getModel(name) {{
  return ALL_MODELS.find(m => m.name === name);
}}

function heroCard(label, value, sub='') {{
  return `<div class="metric-card p-4">
    <div class="section-header mb-1">${{label}}</div>
    <div class="text-2xl font-bold mono text-white">${{value}}</div>
    ${{sub ? `<div class="text-xs text-gray-500 mt-1">${{sub}}</div>` : ''}}
  </div>`;
}}

function renderModelSelector() {{
  const sel = document.getElementById('modelSelector');
  const btns = ALL_MODELS.map((m, i) => {{
    const active = m.name === selectedModel && !compareMode ? 'active' : '';
    return `<button class="model-btn ${{active}} px-3 py-1 rounded text-sm font-medium text-gray-400"
      onclick="selectModel('${{m.name}}')">${{m.name}}</button>`;
  }}).join('');
  sel.innerHTML = `<span class="section-header mr-2">Model</span>` + btns;
}}

function selectModel(name) {{
  selectedModel = name;
  compareMode = false;
  document.getElementById('compareToggle').classList.remove('active');
  renderModelSelector();
  renderAll();
}}

function toggleCompare() {{
  compareMode = !compareMode;
  document.getElementById('compareToggle').classList.toggle('active', compareMode);
  renderModelSelector();
  renderAll();
}}

function switchTab(tab) {{
  activeTab = tab;
  ['overview','commander','oversight','config'].forEach(t => {{
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
  }});
  document.querySelectorAll('.tab-btn').forEach((b, i) => {{
    const tabs = ['overview','commander','oversight','config'];
    b.classList.toggle('active', tabs[i] === tab);
    b.classList.toggle('text-gray-400', tabs[i] !== tab);
  }});
  renderAll();
}}

function renderAll() {{
  if (activeTab === 'overview') renderOverview();
  if (activeTab === 'commander') renderRole('commander');
  if (activeTab === 'oversight') renderRole('oversight');
  if (activeTab === 'config') renderConfig();
}}

function renderOverview() {{
  const models = compareMode ? ALL_MODELS : [getModel(selectedModel)].filter(Boolean);
  if (!models.length) return;

  // Hero metrics for selected/first model
  const m = models[0];
  const cmdRewards = m.commander?.rewards || [];
  const ovRewards = m.oversight?.rewards || [];
  const cmdLast = cmdRewards.length ? cmdRewards[cmdRewards.length-1].toFixed(4) : 'N/A';
  const ovLast = ovRewards.length ? ovRewards[ovRewards.length-1].toFixed(4) : 'N/A';
  const cmdBest = cmdRewards.length ? Math.max(...cmdRewards).toFixed(4) : 'N/A';
  const ovBest = ovRewards.length ? Math.max(...ovRewards).toFixed(4) : 'N/A';

  document.getElementById('heroMetrics').innerHTML =
    heroCard('Commander Final Reward', cmdLast, `Best: ${{cmdBest}} over ${{cmdRewards.length}} steps`) +
    heroCard('Oversight Final Reward', ovLast, `Best: ${{ovBest}} over ${{ovRewards.length}} steps`) +
    heroCard('Commander Steps', cmdRewards.length || 'N/A', m.commander?.config?.base_model?.split('/').pop() || '') +
    heroCard('Oversight Steps', ovRewards.length || 'N/A', m.oversight?.config?.base_model?.split('/').pop() || '');

  // Commander chart
  const cmdDatasets = models.map((mod, i) => {{
    const r = mod.commander?.rewards || [];
    const s = mod.commander?.steps || r.map((_,j) => j+1);
    return lineDs(mod.name + ' Commander', r, COLORS[i % COLORS.length]);
  }});
  const cmdSteps = models[0].commander?.steps || [];
  makeChart('overviewCommanderChart', cmdSteps, cmdDatasets);

  // Oversight chart
  const ovDatasets = models.map((mod, i) => {{
    const r = mod.oversight?.rewards || [];
    return lineDs(mod.name + ' Oversight', r, COLORS[(i+1) % COLORS.length]);
  }});
  const ovSteps = models[0].oversight?.steps || [];
  makeChart('overviewOversightChart', ovSteps, ovDatasets);
}}

function renderRole(role) {{
  const models = compareMode ? ALL_MODELS : [getModel(selectedModel)].filter(Boolean);
  if (!models.length) return;

  const prefix = role === 'commander' ? 'cmd' : 'ov';
  const metricsEl = document.getElementById(prefix + 'Metrics');

  const m = models[0];
  const d = m[role] || {{}};
  const rewards = d.rewards || [];
  const last = rewards.length ? rewards[rewards.length-1].toFixed(4) : 'N/A';
  const best = rewards.length ? Math.max(...rewards).toFixed(4) : 'N/A';
  const fzLast = d.frac_zeros?.length ? d.frac_zeros[d.frac_zeros.length-1].toFixed(2) : 'N/A';

  metricsEl.innerHTML =
    heroCard('Final Reward', last, `${{rewards.length}} steps logged`) +
    heroCard('Best Reward', best, '') +
    heroCard('Last Frac Zero Std', fzLast, '0.00 = all rollouts diverse');

  // Reward
  makeChart(prefix + 'RewardChart',
    models[0][role]?.steps || [],
    models.map((mod, i) => lineDs(mod.name, mod[role]?.rewards || [], COLORS[i % COLORS.length], true))
  );

  // Grad norm
  makeChart(prefix + 'GradChart',
    models[0][role]?.steps || [],
    models.map((mod, i) => lineDs(mod.name, mod[role]?.grad_norms || [], COLORS[i % COLORS.length]))
  );

  // Std
  makeChart(prefix + 'StdChart',
    models[0][role]?.steps || [],
    models.map((mod, i) => lineDs(mod.name, mod[role]?.stds || [], COLORS[i % COLORS.length]))
  );

  // Frac zero
  makeChart(prefix + 'FracChart',
    models[0][role]?.steps || [],
    models.map((mod, i) => lineDs(mod.name, mod[role]?.frac_zeros || [], COLORS[i % COLORS.length]))
  );
}}

function renderConfig() {{
  const models = compareMode ? ALL_MODELS : [getModel(selectedModel)].filter(Boolean);
  const html = models.map(m => {{
    const roles = ['commander', 'oversight'].map(role => {{
      const cfg = m[role]?.config || {{}};
      if (!Object.keys(cfg).length) return '';
      return `
        <div class="mb-3">
          <div class="section-header mb-2">${{role}}</div>
          <table class="w-full text-sm">
            <tr><td class="text-gray-500 pr-4 py-0.5">Base model</td><td class="mono text-xs text-green-400">${{cfg.base_model || 'N/A'}}</td></tr>
            <tr><td class="text-gray-500 pr-4 py-0.5">LoRA r</td><td class="mono text-xs">${{cfg.lora_r || 'N/A'}}</td></tr>
            <tr><td class="text-gray-500 pr-4 py-0.5">LoRA alpha</td><td class="mono text-xs">${{cfg.lora_alpha || 'N/A'}}</td></tr>
            <tr><td class="text-gray-500 pr-4 py-0.5">PEFT version</td><td class="mono text-xs">${{cfg.peft_version || 'N/A'}}</td></tr>
            <tr><td class="text-gray-500 pr-4 py-0.5 align-top">Target modules</td>
              <td class="mono text-xs text-blue-400">${{(cfg.target_modules || []).join(', ')}}</td></tr>
          </table>
        </div>`;
    }}).join('');
    return `<div class="metric-card p-4">
      <div class="text-base font-semibold text-white mb-3">${{m.name}}</div>
      ${{roles}}
    </div>`;
  }}).join('');
  document.getElementById('configPanel').innerHTML = html;
}}

// Init
if (ALL_MODELS.length === 0) {{
  document.getElementById('mainContent').innerHTML = `
    <div class="flex flex-col items-center justify-center h-64 text-gray-500">
      <div class="text-4xl mb-4">📭</div>
      <div class="text-lg font-medium mb-2">No training data found</div>
      <div class="text-sm">Run training and place results in <span class="mono text-blue-400">checkpoints/&lt;model-name&gt;/</span></div>
    </div>`;
}} else {{
  renderModelSelector();
  renderAll();
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    models = scan_checkpoints()
    html = build_html(models)
    OUTPUT_PATH.write_text(html)
    print(f"[training_dashboard] found {len(models)} model(s): {[m['name'] for m in models]}")
    for m in models:
        for role in ("commander", "oversight"):
            d = m.get(role)
            if d and d["rewards"]:
                print(f"  {m['name']}/{role}: {len(d['rewards'])} steps, last reward={d['rewards'][-1]:.4f}")
    print(f"[training_dashboard] written → {OUTPUT_PATH}")
