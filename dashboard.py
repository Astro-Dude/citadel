"""
Citadel — Dashboard Generator

Scans runs/<timestamp>/<task_id>/transcript.json and emits a single
self-contained HTML file that renders:

    * Overview   — score cards + sub-score breakdown per task
    * Timeline   — horizontal step cards (Commander → Oversight → outcome)
    * Charts     — cumulative reward, trust evolution, sub-score radar,
                   decision heatmap
    * Playbook   — shared lesson cards with utility / citation counts
    * Compare    — overlay runs on the same axes (e.g. untrained vs trained)

No backend required — `dashboard.html` embeds all transcript JSON and uses
Chart.js + Tailwind from CDNs. Open it in any browser.

Usage:
    python dashboard.py                      # writes Citadel/runs/dashboard.html
    python dashboard.py --out custom.html    # custom output path
    python dashboard.py --runs-dir /some/path
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Transcript collection
# ---------------------------------------------------------------------------

def collect_runs(runs_root: Path) -> List[Dict[str, Any]]:
    """
    Walk runs/<timestamp>/<task_id>/transcript.json and return a list of
    run dicts: {run_id, summary, tasks: [{task_id, transcript}, ...]}.
    """
    runs: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return runs

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        tasks = []
        for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            tj = task_dir / "transcript.json"
            if not tj.exists():
                continue
            try:
                with open(tj) as f:
                    transcript = json.load(f)
            except Exception as e:
                print(f"[WARN] failed to read {tj}: {e}")
                continue
            tasks.append({"task_id": task_dir.name, "transcript": transcript})
        if not tasks:
            continue

        summary = {}
        summary_json = run_dir / "summary.json"
        if summary_json.exists():
            try:
                with open(summary_json) as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        runs.append({
            "run_id": run_dir.name,
            "summary": summary,
            "tasks": tasks,
        })
    return runs


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Citadel — Council Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    html, body { height: 100%; background: #0a0a0e; color: #e5e7eb; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0f0f14; }
    ::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 5px; }
    .glow-red { box-shadow: 0 0 14px 0 rgba(239,68,68,.35); }
    .glow-green { box-shadow: 0 0 14px 0 rgba(34,197,94,.30); }
    .glow-amber { box-shadow: 0 0 14px 0 rgba(245,158,11,.30); }
    .glow-blue { box-shadow: 0 0 14px 0 rgba(59,130,246,.30); }
    .fade-enter { animation: fade .2s ease-in; }
    @keyframes fade { from {opacity:0; transform: translateY(4px);} to {opacity:1; transform:none;} }
    .card { background: linear-gradient(180deg, #12121a 0%, #0f0f17 100%); border: 1px solid #1f1f2a; }
    .tab-btn.active { background: #4c1d95; color: white; }
    .timeline-card { min-width: 320px; max-width: 360px; }
    details summary { cursor: pointer; user-select: none; }
    details summary::-webkit-details-marker { display: none; }
    details[open] > summary .chev { transform: rotate(90deg); }
    .chev { transition: transform .15s ease; }
    pre.code { white-space: pre-wrap; word-break: break-word; background: #0b0b12; border: 1px solid #1a1a24; padding: 10px; border-radius: 6px; font-size: 11px; line-height: 1.45; color: #d1d5db; max-height: 320px; overflow-y: auto; }
    .score-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .pill { display: inline-block; padding: 1px 8px; border-radius: 9999px; font-size: 10px; font-weight: 600; letter-spacing: .02em; }
  </style>
</head>
<body class="min-h-screen">
  <!-- ================= HEADER ================= -->
  <header class="border-b border-violet-900/40 bg-gradient-to-r from-violet-950/60 via-indigo-950/40 to-rose-950/30 backdrop-blur">
    <div class="max-w-[1700px] mx-auto px-6 py-4 flex items-center gap-6">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded bg-violet-600 flex items-center justify-center text-white font-bold">🏰</div>
        <div>
          <div class="text-xl font-bold tracking-tight">Citadel Council Dashboard</div>
          <div class="text-xs text-violet-300/70">Multi-agent AI defense — Commander × Oversight × Governance × Trust</div>
        </div>
      </div>
      <div class="ml-auto flex items-center gap-2 text-xs text-violet-300/70">
        <span class="pill bg-violet-900/60 text-violet-200">Round 2 / OpenEnv</span>
        <span id="run-count" class="pill bg-indigo-900/60 text-indigo-200">— runs</span>
      </div>
    </div>
  </header>

  <div class="max-w-[1700px] mx-auto px-6 py-6 grid gap-6" style="grid-template-columns: 280px 1fr">
    <!-- ================ SIDEBAR ================ -->
    <aside class="card rounded-xl p-4 h-fit sticky top-4">
      <div class="text-[11px] uppercase tracking-widest text-violet-300/60 mb-3">Runs</div>
      <div id="runs-list" class="space-y-1"></div>
      <div class="mt-5 text-[11px] uppercase tracking-widest text-violet-300/60 mb-2">Tabs</div>
      <div class="flex flex-wrap gap-1">
        <button class="tab-btn px-3 py-1.5 rounded text-xs hover:bg-violet-900/40" data-tab="overview">Overview</button>
        <button class="tab-btn px-3 py-1.5 rounded text-xs hover:bg-violet-900/40" data-tab="timeline">Timeline</button>
        <button class="tab-btn px-3 py-1.5 rounded text-xs hover:bg-violet-900/40" data-tab="charts">Charts</button>
        <button class="tab-btn px-3 py-1.5 rounded text-xs hover:bg-violet-900/40" data-tab="playbook">Playbook</button>
        <button class="tab-btn px-3 py-1.5 rounded text-xs hover:bg-violet-900/40" data-tab="compare">Compare</button>
      </div>
    </aside>

    <!-- ================ MAIN ================ -->
    <main id="main">
      <!-- Filled in by JS -->
    </main>
  </div>

  <script id="citadel-data" type="application/json">__DATA_JSON__</script>

  <script>
    // ============================================================
    // Data + state
    // ============================================================
    const DATA = JSON.parse(document.getElementById("citadel-data").textContent);
    const STATE = {
      runId: null,
      taskId: null,
      tab: "overview",
      charts: {},
      compareSel: new Set(),
    };

    const DECISION_COLORS = {
      APPROVE: { bg: "bg-emerald-900/40", ring: "ring-emerald-600/60", text: "text-emerald-300", glow: "glow-green", label: "APPROVE" },
      REVISE:  { bg: "bg-amber-900/40",   ring: "ring-amber-600/60",   text: "text-amber-300",   glow: "glow-amber", label: "REVISE" },
      VETO:    { bg: "bg-rose-900/40",    ring: "ring-rose-600/60",    text: "text-rose-300",    glow: "glow-red",   label: "VETO" },
      FLAG_FOR_HUMAN: { bg: "bg-sky-900/40", ring: "ring-sky-600/60",  text: "text-sky-300",     glow: "glow-blue",  label: "FLAG" },
    };

    // ============================================================
    // Helpers
    // ============================================================
    function $(sel, root=document) { return root.querySelector(sel); }
    function $$(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }
    function el(tag, attrs={}, children=[]) {
      const e = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") e.className = v;
        else if (k === "html") e.innerHTML = v;
        else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
        else e.setAttribute(k, v);
      }
      for (const c of (Array.isArray(children) ? children : [children])) {
        if (c == null) continue;
        e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
      return e;
    }
    function fmtPct(v) { return (v == null || isNaN(v)) ? "—" : (v * 100).toFixed(0) + "%"; }
    function fmtNum(v, d=3) { return (v == null || isNaN(v)) ? "—" : Number(v).toFixed(d); }
    function getRun(runId) { return DATA.runs.find(r => r.run_id === runId); }
    function getTask(runId, taskId) { return getRun(runId)?.tasks.find(t => t.task_id === taskId); }
    function clearCharts() {
      for (const k of Object.keys(STATE.charts)) {
        try { STATE.charts[k].destroy(); } catch (_) {}
        delete STATE.charts[k];
      }
    }

    // ============================================================
    // Sidebar
    // ============================================================
    function renderSidebar() {
      const wrap = $("#runs-list");
      wrap.innerHTML = "";
      for (const run of DATA.runs) {
        const rt = el("div", { class: "mb-2" });
        const label = el("div", {
          class: "font-semibold text-xs text-violet-200 truncate",
          title: run.run_id,
        }, run.run_id);
        rt.appendChild(label);
        for (const task of run.tasks) {
          const score = task.transcript?.final_scores?.final_score ?? task.transcript?.reported_score ?? 0;
          const cat = task.transcript?.final_scores?.catastrophic ? "☠" : "";
          const tl = el("button", {
            class: "w-full text-left text-xs px-2 py-1 rounded hover:bg-violet-900/40 truncate",
            onclick: () => { STATE.runId = run.run_id; STATE.taskId = task.task_id; render(); },
          }, `${task.task_id} — ${(score*100).toFixed(0)}% ${cat}`);
          if (STATE.runId === run.run_id && STATE.taskId === task.task_id) {
            tl.classList.add("bg-violet-900/60", "text-white");
          }
          rt.appendChild(tl);
        }
        wrap.appendChild(rt);
      }
      $("#run-count").textContent = `${DATA.runs.length} run${DATA.runs.length === 1 ? "" : "s"}`;
      // Active tab button
      $$("#main, aside button.tab-btn").forEach(()=>{});
      $$(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === STATE.tab);
      });
    }

    // ============================================================
    // Main router
    // ============================================================
    function render() {
      clearCharts();
      const main = $("#main");
      main.innerHTML = "";
      main.classList.add("fade-enter");
      setTimeout(() => main.classList.remove("fade-enter"), 250);

      if (!STATE.runId || !STATE.taskId) {
        // Pick defaults
        if (DATA.runs.length > 0) {
          STATE.runId = DATA.runs[0].run_id;
          STATE.taskId = DATA.runs[0].tasks[0]?.task_id || null;
        }
      }
      renderSidebar();
      if (!STATE.runId) {
        main.appendChild(emptyState());
        return;
      }

      switch (STATE.tab) {
        case "overview":  renderOverview(main); break;
        case "timeline":  renderTimeline(main); break;
        case "charts":    renderCharts(main); break;
        case "playbook":  renderPlaybook(main); break;
        case "compare":   renderCompare(main); break;
      }
    }

    function emptyState() {
      return el("div", { class: "card rounded-xl p-10 text-center" }, [
        el("div", { class: "text-2xl mb-2" }, "🏰"),
        el("div", { class: "text-lg font-semibold" }, "No runs found yet"),
        el("div", { class: "text-sm text-gray-400 mt-2" }, "Run inference.py to generate transcripts, then regenerate the dashboard."),
      ]);
    }

    // ============================================================
    // Overview
    // ============================================================
    function renderOverview(root) {
      const run = getRun(STATE.runId);
      if (!run) return;

      // Run-level header
      const hdr = el("div", { class: "card rounded-xl p-5 mb-5" }, [
        el("div", { class: "flex items-start justify-between" }, [
          el("div", {}, [
            el("div", { class: "text-[11px] uppercase tracking-widest text-violet-300/60" }, "Run"),
            el("div", { class: "text-xl font-bold" }, run.run_id),
            el("div", { class: "text-xs text-gray-400 mt-1" },
              `Model: ${run.summary?.model_name || run.tasks[0]?.transcript?.model_name || "?"} · ${run.tasks.length} task(s) · avg score ${(run.summary?.average_score ?? 0).toFixed(3)}`
            ),
          ]),
          el("div", { class: "text-right" }, [
            el("div", { class: "text-4xl font-black text-violet-300" }, fmtPct(run.summary?.average_score || 0)),
            el("div", { class: "text-[11px] uppercase text-violet-300/60" }, "Avg final_score"),
          ]),
        ])
      ]);
      root.appendChild(hdr);

      // Task cards grid
      const grid = el("div", { class: "grid gap-4", style: "grid-template-columns: repeat(auto-fit, minmax(380px, 1fr))" });
      for (const task of run.tasks) {
        const t = task.transcript;
        const fs = t.final_scores || {};
        const isCat = fs.catastrophic;
        const selected = task.task_id === STATE.taskId;

        const subrows = [
          ["bastion core",           fs.bastion_v1_final_score, "#818cf8"],
          ["governance",             fs.governance_compliance, "#34d399"],
          ["oversight precision",    fs.oversight_precision,   "#f472b6"],
          ["trust maintenance",      fs.trust_maintenance,     "#fbbf24"],
          ["efficiency",             fs.efficiency,            "#38bdf8"],
          ["adversary adaptation",   fs.adversary_adaptation,  "#a78bfa"],
        ];

        const bars = subrows.map(([lbl, v, color]) => {
          const pct = Math.max(0, Math.min(100, (Number(v) || 0) * 100));
          return el("div", { class: "flex items-center gap-3 text-xs" }, [
            el("div", { class: "w-36 text-gray-400" }, lbl),
            el("div", { class: "flex-1 h-2 bg-gray-800 rounded overflow-hidden" }, [
              el("div", { class: "h-full", style: `width:${pct}%; background:${color}` }),
            ]),
            el("div", { class: "w-12 text-right font-mono text-gray-300" }, pct.toFixed(0) + "%"),
          ]);
        });

        const card = el("div", {
          class: `card rounded-xl p-5 cursor-pointer hover:border-violet-600 transition ${selected ? "ring-2 ring-violet-500" : ""}`,
          onclick: () => { STATE.runId = run.run_id; STATE.taskId = task.task_id; STATE.tab = "timeline"; render(); },
        }, [
          el("div", { class: "flex items-start justify-between mb-3" }, [
            el("div", {}, [
              el("div", { class: "text-xs text-violet-300/70" }, `Gen ${t.adversary_gen} · ${(t.duration_s || 0).toFixed(0)}s`),
              el("div", { class: "text-lg font-bold" }, task.task_id),
              el("div", { class: "text-[11px] text-gray-500 mt-0.5" }, t.termination_reason || ""),
            ]),
            el("div", { class: "text-right" }, [
              el("div", { class: `text-3xl font-black ${isCat ? "text-rose-400" : "text-emerald-300"}` }, fmtPct(fs.final_score || 0)),
              el("div", { class: "text-[11px] uppercase text-gray-500" }, "final"),
              isCat ? el("div", { class: "pill bg-rose-900 text-rose-200 mt-1" }, "☠ CATASTROPHIC") : null,
            ]),
          ]),
          el("div", { class: "space-y-2 mt-4" }, bars),
          el("div", { class: "grid grid-cols-4 gap-2 mt-4 text-[10px]" }, [
            statChip("proposals", t.council_summary?.total_proposals),
            statChip("vetoes", `${t.council_summary?.correct_vetoes || 0}/${t.council_summary?.vetoes || 0}`),
            statChip("revisions", t.council_summary?.revisions),
            statChip("violations", t.governance_final?.violations_count),
          ]),
        ]);
        grid.appendChild(card);
      }
      root.appendChild(grid);
    }

    function statChip(label, value) {
      return el("div", { class: "bg-gray-900/70 rounded px-2 py-1 text-center" }, [
        el("div", { class: "text-gray-500 uppercase tracking-wider" }, label),
        el("div", { class: "text-sm font-bold text-gray-200" }, String(value ?? "—")),
      ]);
    }

    // ============================================================
    // Timeline
    // ============================================================
    function renderTimeline(root) {
      const task = getTask(STATE.runId, STATE.taskId);
      if (!task) return;
      const t = task.transcript;

      // Summary strip
      root.appendChild(el("div", { class: "card rounded-xl p-4 mb-4 flex items-center justify-between" }, [
        el("div", {}, [
          el("div", { class: "text-xs text-violet-300/60 uppercase tracking-wider" }, "Timeline"),
          el("div", { class: "text-lg font-bold" }, `${t.task_id} · ${t.steps.length} steps · Gen ${t.adversary_gen}`),
        ]),
        el("div", { class: "flex gap-3 text-xs" }, [
          timelineStat("Final", fmtPct(t.final_scores?.final_score), "text-emerald-300"),
          timelineStat("Bastion core", fmtPct(t.final_scores?.bastion_v1_final_score), "text-indigo-300"),
          timelineStat("Governance", fmtPct(t.final_scores?.governance_compliance), "text-green-300"),
          timelineStat("Trust maint.", fmtPct(t.final_scores?.trust_maintenance), "text-amber-300"),
        ]),
      ]));

      const track = el("div", { class: "flex gap-3 overflow-x-auto pb-3" });
      for (const step of t.steps) {
        track.appendChild(stepCard(step));
      }
      root.appendChild(track);

      // Forensic report block
      if (t.forensic_report && Object.keys(t.forensic_report).length) {
        root.appendChild(collapsible("🔎 Forensic Report (episode end)", renderForensic(t.forensic_report)));
      }
    }

    function timelineStat(label, val, color="text-gray-200") {
      return el("div", { class: "text-center" }, [
        el("div", { class: "text-[10px] uppercase text-gray-500" }, label),
        el("div", { class: `text-sm font-bold ${color}` }, val),
      ]);
    }

    function stepCard(step) {
      const ov = step.oversight?.parsed || {};
      const decision = ov.decision_name || "APPROVE";
      const dec = DECISION_COLORS[decision] || DECISION_COLORS.APPROVE;
      const cmd = step.commander?.parsed || {};
      const rev = step.revision?.parsed;
      const envr = step.env_result || {};
      const rw = envr.commander_reward ?? 0;

      const rewardColor = rw > 0.05 ? "text-emerald-400" : rw < -0.05 ? "text-rose-400" : "text-gray-400";

      const card = el("div", {
        class: `timeline-card card rounded-xl p-3 ring-1 ${dec.ring} ${dec.glow}`,
      }, [
        el("div", { class: "flex items-center justify-between text-[10px] uppercase tracking-widest mb-2" }, [
          el("div", { class: "text-gray-500" }, `Step ${step.step} · hr ${step.hour}`),
          el("div", { class: `pill ${dec.bg} ${dec.text}` }, dec.label + (rev ? " + REVISED" : "")),
        ]),
        el("div", { class: "text-xs mb-2" }, [
          el("div", { class: "text-gray-400" }, "Commander proposed:"),
          el("div", { class: "font-semibold text-gray-100 truncate" },
            `${cmd.action_name_initial || "?"} → target ${cmd.target_system ?? "?"}`
          ),
          el("div", { class: "text-[11px] text-gray-500 mt-1 italic line-clamp-3" },
            (cmd.justification || "").slice(0, 200) || "(no justification)"
          ),
        ]),
        el("div", { class: "text-xs mb-2" }, [
          el("div", { class: "text-gray-400" }, `Oversight ${dec.label.toLowerCase()} (risk ${ov.risk_tier || 1})`),
          ov.weakness ? el("div", { class: "text-[11px] text-gray-300 mt-1 line-clamp-3" }, `"${ov.weakness}"`) : null,
          ov.counter_proposal ? el("div", { class: "text-[11px] text-sky-300 mt-1" },
            `→ counter: ${ov.counter_proposal.action}/${ov.counter_proposal.target_system}`) : null,
        ]),
        rev ? el("div", { class: "text-xs mb-2 bg-amber-950/40 rounded px-2 py-1" }, [
          el("div", { class: "text-amber-300" }, `Revised → ${rev.action_name_final || "?"} / tgt ${rev.target_system ?? "?"}`),
          el("div", { class: "text-[11px] text-amber-200 italic line-clamp-2" }, (rev.justification || "").slice(0, 180)),
        ]) : null,
        ov.lesson_text ? el("div", { class: "text-[10px] bg-violet-950/40 rounded px-2 py-1 mt-2 text-violet-200" }, `📖 lesson: ${ov.lesson_text}`) : null,
        el("div", { class: "flex items-center justify-between mt-3 text-[11px]" }, [
          el("div", { class: `font-mono ${rewardColor}` }, (rw >= 0 ? "+" : "") + Number(rw).toFixed(3)),
          el("div", { class: "text-gray-500" }, envr.info?.applied === false ? "not applied" : "applied"),
        ]),
        el("details", { class: "mt-2 text-[10px] text-gray-500" }, [
          el("summary", {}, [el("span", { class: "chev inline-block mr-1" }, "▸"), "raw prompts + responses"]),
          el("div", { class: "mt-2" }, [
            el("div", { class: "text-gray-400 mt-2" }, "Commander raw response:"),
            el("pre", { class: "code" }, step.commander?.raw_response || ""),
            el("div", { class: "text-gray-400 mt-2" }, "Oversight raw response:"),
            el("pre", { class: "code" }, step.oversight?.raw_response || ""),
            rev ? el("div", { class: "text-gray-400 mt-2" }, "Revision raw response:") : null,
            rev ? el("pre", { class: "code" }, step.revision?.raw_response || "") : null,
          ]),
        ]),
      ]);
      return card;
    }

    function collapsible(title, content) {
      const wrap = el("details", { class: "card rounded-xl p-4 mt-4" }, [
        el("summary", { class: "text-sm font-semibold flex items-center gap-2" },
          [el("span", { class: "chev" }, "▸"), title]),
        el("div", { class: "mt-3" }, content),
      ]);
      return wrap;
    }

    function renderForensic(f) {
      const pieces = [];
      if (f.incident_summary) {
        const rows = Object.entries(f.incident_summary).map(([k, v]) =>
          el("div", { class: "flex justify-between text-xs py-1 border-b border-gray-800" }, [
            el("div", { class: "text-gray-500" }, k),
            el("div", { class: "font-mono" }, String(v)),
          ])
        );
        pieces.push(el("div", { class: "mb-3" }, rows));
      }
      if (f.grades) {
        const rows = Object.entries(f.grades).map(([k, v]) => {
          const grade = v.grade || "—";
          const colorMap = { A: "text-emerald-300", B: "text-lime-300", C: "text-amber-300", D: "text-orange-300", F: "text-rose-400" };
          return el("div", { class: "flex justify-between text-xs py-1 border-b border-gray-800" }, [
            el("div", { class: "text-gray-500" }, k),
            el("div", { class: `font-mono font-bold ${colorMap[grade] || "text-gray-300"}` }, `${grade} (${v.score})`),
          ]);
        });
        pieces.push(el("div", { class: "mb-3" }, rows));
      }
      if (f.recommendations?.length) {
        pieces.push(el("div", { class: "text-xs text-gray-400 mb-1" }, "Recommendations:"));
        pieces.push(el("ul", { class: "list-disc list-inside text-[11px] text-gray-300 space-y-1" },
          f.recommendations.map(r => el("li", {}, r))));
      }
      return el("div", {}, pieces);
    }

    // ============================================================
    // Charts
    // ============================================================
    function renderCharts(root) {
      const task = getTask(STATE.runId, STATE.taskId);
      if (!task) return;
      const t = task.transcript;

      const grid = el("div", { class: "grid gap-4", style: "grid-template-columns: 1fr 1fr" });

      // Cumulative reward
      const rewardCard = chartCard("Cumulative Commander reward", "chart-reward");
      const trustCard = chartCard("Trust evolution (C→O + O→C)", "chart-trust");
      const radarCard = chartCard("Sub-score radar", "chart-radar");
      const decisionCard = chartCard("Decision outcomes", "chart-decisions");

      grid.appendChild(rewardCard);
      grid.appendChild(trustCard);
      grid.appendChild(radarCard);
      grid.appendChild(decisionCard);
      root.appendChild(grid);

      // Populate charts after DOM insertion
      const steps = t.steps || [];
      const stepNums = steps.map(s => `s${s.step}`);
      let cum = 0;
      const cumulative = steps.map(s => (cum += s.env_result?.commander_reward || 0));

      STATE.charts.reward = new Chart($("#chart-reward").getContext("2d"), {
        type: "line",
        data: {
          labels: stepNums,
          datasets: [{ label: "cumulative reward", data: cumulative, borderColor: "#a78bfa", backgroundColor: "rgba(167,139,250,.2)", fill: true, tension: 0.3 }],
        },
        options: chartOpts(),
      });

      // Trust — read trust_final.history (if available from state) else per-step trust_after
      const tc2o = [], to2c = [];
      for (const s of steps) {
        const ta = s.env_result?.trust_after || {};
        tc2o.push(ta.trust_commander_in_oversight ?? null);
        to2c.push(ta.trust_oversight_in_commander ?? null);
      }
      STATE.charts.trust = new Chart($("#chart-trust").getContext("2d"), {
        type: "line",
        data: {
          labels: stepNums,
          datasets: [
            { label: "Commander → Oversight", data: tc2o, borderColor: "#f472b6", tension: 0.3 },
            { label: "Oversight → Commander", data: to2c, borderColor: "#60a5fa", tension: 0.3 },
          ],
        },
        options: {...chartOpts(), scales: { y: { min: 0, max: 1, ticks: {color:"#9ca3af"} }, x: { ticks:{color:"#9ca3af"}}}},
      });

      // Radar of sub-scores
      const fs = t.final_scores || {};
      STATE.charts.radar = new Chart($("#chart-radar").getContext("2d"), {
        type: "radar",
        data: {
          labels: ["bastion core", "governance", "veto prec.", "trust maint.", "efficiency", "adv. adapt."],
          datasets: [{
            label: task.task_id,
            data: [fs.bastion_v1_final_score, fs.governance_compliance, fs.oversight_precision, fs.trust_maintenance, fs.efficiency, fs.adversary_adaptation].map(x => Number(x) || 0),
            backgroundColor: "rgba(167,139,250,.25)", borderColor: "#a78bfa",
          }],
        },
        options: {...chartOpts(), scales: { r: { min: 0, max: 1, angleLines: {color:"#2a2a38"}, grid:{color:"#2a2a38"}, pointLabels:{color:"#cbd5e1", font:{size:11}}, ticks:{color:"#6b7280", backdropColor:"transparent"}}}},
      });

      // Decision outcomes bar
      const cs = t.council_summary || {};
      STATE.charts.decisions = new Chart($("#chart-decisions").getContext("2d"), {
        type: "bar",
        data: {
          labels: ["approvals", "revisions", "correct vetoes", "false vetoes", "flags"],
          datasets: [{
            data: [cs.approvals||0, cs.revisions||0, cs.correct_vetoes||0, cs.false_vetoes||0, cs.flags||0],
            backgroundColor: ["#34d399", "#fbbf24", "#60a5fa", "#f43f5e", "#a78bfa"],
          }],
        },
        options: {...chartOpts(), plugins: { legend: { display: false } }},
      });
    }

    function chartCard(title, canvasId) {
      return el("div", { class: "card rounded-xl p-4" }, [
        el("div", { class: "text-xs uppercase tracking-widest text-violet-300/70 mb-3" }, title),
        el("canvas", { id: canvasId, style: "max-height: 260px" }),
      ]);
    }
    function chartOpts() {
      return {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
        scales: { x: { ticks: { color: "#9ca3af" }, grid: { color: "#1a1a24" } }, y: { ticks: { color: "#9ca3af" }, grid: { color: "#1a1a24" } } },
      };
    }

    // ============================================================
    // Playbook
    // ============================================================
    function renderPlaybook(root) {
      // Collect all lessons from this run's tasks (from oversight.parsed.lesson_text per step)
      const run = getRun(STATE.runId);
      if (!run) return;
      const lessons = [];
      for (const task of run.tasks) {
        for (const s of (task.transcript.steps || [])) {
          const lt = s.oversight?.parsed?.lesson_text;
          if (lt && lt.trim()) {
            lessons.push({ text: lt, tags: s.oversight?.parsed?.lesson_tags || [], task: task.task_id, step: s.step });
          }
        }
      }
      if (lessons.length === 0) {
        root.appendChild(el("div", { class: "card rounded-xl p-6 text-center text-gray-400" },
          "Oversight didn't write any lessons in this run — the playbook is empty."));
        return;
      }
      const grid = el("div", { class: "grid gap-3", style: "grid-template-columns: repeat(auto-fill, minmax(340px,1fr))" });
      for (const l of lessons) {
        grid.appendChild(el("div", { class: "card rounded-xl p-4" }, [
          el("div", { class: "text-[10px] uppercase tracking-widest text-violet-300/60 mb-1" }, `${l.task} · step ${l.step}`),
          el("div", { class: "text-sm text-gray-100 leading-snug" }, l.text),
          el("div", { class: "mt-2 flex flex-wrap gap-1" },
            (l.tags || []).map(t => el("span", { class: "pill bg-violet-900/60 text-violet-200" }, t))),
        ]));
      }
      root.appendChild(grid);
    }

    // ============================================================
    // Compare
    // ============================================================
    function renderCompare(root) {
      // Pick all (run, task) pairs, let user toggle, then overlay trust + reward.
      const wrap = el("div", { class: "space-y-4" });
      const selectorsCard = el("div", { class: "card rounded-xl p-4" }, [
        el("div", { class: "text-xs uppercase tracking-widest text-violet-300/70 mb-2" }, "Select runs/tasks to overlay"),
      ]);
      const selectors = el("div", { class: "flex flex-wrap gap-2" });
      for (const run of DATA.runs) {
        for (const task of run.tasks) {
          const key = `${run.run_id}::${task.task_id}`;
          const active = STATE.compareSel.has(key);
          selectors.appendChild(el("button", {
            class: `pill ${active ? "bg-violet-700 text-white" : "bg-gray-800 text-gray-300"}`,
            onclick: () => {
              if (active) STATE.compareSel.delete(key); else STATE.compareSel.add(key);
              render();
            },
          }, `${run.run_id.slice(0, 24)} · ${task.task_id}`));
        }
      }
      selectorsCard.appendChild(selectors);
      wrap.appendChild(selectorsCard);

      if (STATE.compareSel.size === 0) {
        wrap.appendChild(el("div", { class: "card rounded-xl p-6 text-center text-gray-400" }, "Pick two or more tasks above to overlay their trust + reward."));
        root.appendChild(wrap);
        return;
      }

      const grid = el("div", { class: "grid gap-4", style: "grid-template-columns: 1fr 1fr" });
      grid.appendChild(chartCard("Cumulative reward (overlay)", "cmp-reward"));
      grid.appendChild(chartCard("Trust min(c2o, o2c) (overlay)", "cmp-trust"));
      grid.appendChild(chartCard("Final sub-scores", "cmp-radar"));
      grid.appendChild(chartCard("Comparative bar", "cmp-bar"));
      wrap.appendChild(grid);
      root.appendChild(wrap);

      const selected = Array.from(STATE.compareSel).map(k => {
        const [runId, taskId] = k.split("::");
        return { runId, taskId, transcript: getTask(runId, taskId).transcript };
      });

      const colors = ["#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#60a5fa", "#f87171", "#38bdf8"];
      // Reward overlay
      const rewardData = {
        labels: Array.from({length: Math.max(...selected.map(s => s.transcript.steps.length))}, (_, i) => `s${i+1}`),
        datasets: selected.map((s, i) => {
          let c = 0;
          return { label: `${s.runId.slice(0,12)}/${s.taskId}`, data: s.transcript.steps.map(x => (c += x.env_result?.commander_reward || 0)), borderColor: colors[i % colors.length], tension: 0.3 };
        }),
      };
      STATE.charts.cmpReward = new Chart($("#cmp-reward").getContext("2d"), { type: "line", data: rewardData, options: chartOpts() });

      // Trust overlay
      const trustData = {
        labels: rewardData.labels,
        datasets: selected.map((s, i) => ({
          label: `${s.runId.slice(0,12)}/${s.taskId}`,
          data: s.transcript.steps.map(x => {
            const ta = x.env_result?.trust_after || {};
            const a = ta.trust_commander_in_oversight;
            const b = ta.trust_oversight_in_commander;
            return (a == null || b == null) ? null : Math.min(a, b);
          }),
          borderColor: colors[i % colors.length], tension: 0.3,
        })),
      };
      STATE.charts.cmpTrust = new Chart($("#cmp-trust").getContext("2d"), { type: "line", data: trustData, options: {...chartOpts(), scales: { y: { min: 0, max: 1, ticks:{color:"#9ca3af"} }, x: { ticks:{color:"#9ca3af"}}}}});

      // Radar: average sub-scores
      const labels = ["bastion", "gov.", "veto prec.", "trust", "eff.", "adv."];
      STATE.charts.cmpRadar = new Chart($("#cmp-radar").getContext("2d"), {
        type: "radar",
        data: {
          labels,
          datasets: selected.map((s, i) => {
            const f = s.transcript.final_scores || {};
            return {
              label: `${s.runId.slice(0,10)}/${s.taskId}`,
              data: [f.bastion_v1_final_score, f.governance_compliance, f.oversight_precision, f.trust_maintenance, f.efficiency, f.adversary_adaptation].map(x => Number(x) || 0),
              backgroundColor: colors[i % colors.length] + "40",
              borderColor: colors[i % colors.length],
            };
          }),
        },
        options: {...chartOpts(), scales: { r: { min: 0, max: 1, angleLines: {color:"#2a2a38"}, grid:{color:"#2a2a38"}, pointLabels:{color:"#cbd5e1", font:{size:11}}, ticks:{color:"#6b7280", backdropColor:"transparent"}}}},
      });

      // Bar: final scores
      STATE.charts.cmpBar = new Chart($("#cmp-bar").getContext("2d"), {
        type: "bar",
        data: {
          labels: selected.map(s => `${s.taskId}@${s.runId.slice(0,8)}`),
          datasets: [{ label: "final_score", data: selected.map(s => s.transcript.final_scores?.final_score || 0), backgroundColor: colors }],
        },
        options: {...chartOpts(), scales: { y: { min: 0, max: 1, ticks:{color:"#9ca3af"} }, x: { ticks:{color:"#9ca3af"}}}, plugins: { legend: {display: false}}},
      });
    }

    // ============================================================
    // Wiring
    // ============================================================
    $$(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        STATE.tab = btn.dataset.tab;
        render();
      });
    });

    render();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Generator entry
# ---------------------------------------------------------------------------

def build(runs_dir: Path, out: Path) -> Path:
    runs = collect_runs(runs_dir)
    data = {"runs": runs}
    payload = json.dumps(data, default=str)
    # Escape </script> so it can't break out of the <script> tag
    payload = payload.replace("</script>", "<\\/script>")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Citadel dashboard HTML")
    p.add_argument("--runs-dir", default="runs", help="Root of run transcripts (default: ./runs)")
    p.add_argument("--out", default=None, help="Output HTML path (default: <runs-dir>/dashboard.html)")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    out = Path(args.out).resolve() if args.out else (runs_dir / "dashboard.html")

    path = build(runs_dir, out)
    # Count what we rendered
    runs = collect_runs(runs_dir)
    total_tasks = sum(len(r["tasks"]) for r in runs)
    size_kb = os.path.getsize(path) / 1024
    print(f"dashboard: {path}")
    print(f"  {len(runs)} run(s), {total_tasks} task transcript(s), {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
