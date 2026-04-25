"""
Citadel — Shared Playbook (second Theme 4 hook)

The council (Commander + Oversight) accumulates one-line lessons across
episodes. After each action resolves, Oversight writes a tagged lesson.
Both agents see the current playbook at the start of every future episode.

Lesson lifecycle:
  1. WRITTEN by Oversight after an action outcome is observed
  2. TAGGED with context (adversary_gen, system_type, alert_confidence_band)
  3. SURFACED in both agents' observations on subsequent episodes
  4. CITED when Commander references its lesson_id in a justification
  5. SCORED by lesson_utility:
       +1 per citation that led to an improved outcome
       -1 per citation that led to a worse outcome
       decay toward 0 if never cited
  6. PRUNED when utility drops below floor (or capacity exceeded)

This gives us:
  - a recursive self-improvement loop (lessons improve future episodes)
  - a judge-inspectable artifact (the playbook itself, exportable as text)
  - a trainable "lesson_utility" signal that feeds reward shaping
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lesson data model
# ---------------------------------------------------------------------------

@dataclass
class Lesson:
    lesson_id: str                # e.g. "L-0042"
    text: str                     # one-line human-readable rule
    tags: List[str] = field(default_factory=list)
    # Provenance
    authored_by: str = "oversight"        # always "oversight" for now
    adversary_gen: int = 1                # which adversary generation wrote it
    task_id: str = ""                     # which task it came from
    hour: int = 0                         # step it was written at
    # Utility tracking (updated by env across episodes)
    citations: int = 0                    # times Commander cited it
    wins: int = 0                         # citations followed by improved outcome
    losses: int = 0                       # citations followed by worse outcome
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = 0.0

    @property
    def utility(self) -> float:
        """Net value of this lesson. Range roughly [-1, +1]."""
        total = self.wins + self.losses
        if total == 0:
            # Uncited lessons decay slowly toward 0
            age_hours = (time.time() - self.created_ts) / 3600
            return max(0.0, 0.1 - 0.01 * age_hours)
        return (self.wins - self.losses) / max(1, total)

    def to_obs_dict(self) -> Dict[str, Any]:
        """Compact representation for inclusion in an agent observation."""
        return {
            "id": self.lesson_id,
            "text": self.text,
            "tags": self.tags,
            "utility": round(self.utility, 2),
            "citations": self.citations,
        }


# ---------------------------------------------------------------------------
# Tag helpers — canonical context tags for lesson retrieval
# ---------------------------------------------------------------------------

ADVERSARY_GEN_TAGS = {1: "gen_1_script", 2: "gen_2_adaptive", 3: "gen_3_deceptive"}

SYSTEM_TYPE_TAGS = {
    "database": "data_system",
    "file_server": "data_system",
    "email_server": "data_system",
    "backup_server": "data_system",
    "web_server": "service_system",
    "app_server": "service_system",
    "workstations": "endpoint",
    "firewall": "perimeter",
}


def confidence_band_tag(confidence: float) -> str:
    if confidence < 0.30:
        return "low_confidence_alert"
    if confidence < 0.70:
        return "medium_confidence_alert"
    return "high_confidence_alert"


def make_context_tags(
    adversary_gen: int,
    system_name: str = "",
    alert_confidence: float = -1.0,
    extras: Optional[List[str]] = None,
) -> List[str]:
    tags: List[str] = [ADVERSARY_GEN_TAGS.get(adversary_gen, "gen_1_script")]
    if system_name and system_name in SYSTEM_TYPE_TAGS:
        tags.append(SYSTEM_TYPE_TAGS[system_name])
    if alert_confidence >= 0:
        tags.append(confidence_band_tag(alert_confidence))
    if extras:
        tags.extend(extras)
    return tags


# ---------------------------------------------------------------------------
# Playbook — the council's shared memory
# ---------------------------------------------------------------------------

class Playbook:
    """
    A bounded list of Lessons with retrieval, utility scoring, and decay.

    Persistence: lessons are optionally mirrored to a JSON file on disk so
    they survive across training/eval runs. The file path is configurable
    via the CITADEL_PLAYBOOK_PATH env var (default: ./playbook.json).
    """

    def __init__(
        self,
        capacity: int = 64,
        min_utility: float = -0.5,
        path: Optional[str] = None,
    ) -> None:
        self.capacity = capacity
        self.min_utility = min_utility
        self.path = path or os.getenv("CITADEL_PLAYBOOK_PATH", "./playbook.json")
        self._lessons: List[Lesson] = []
        self._next_id = 1
        self._load_if_exists()

    # --- persistence -------------------------------------------------------

    def _load_if_exists(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            for d in data.get("lessons", []):
                self._lessons.append(Lesson(**d))
            self._next_id = data.get("next_id", len(self._lessons) + 1)
        except Exception:
            # A corrupt playbook file shouldn't break the env
            self._lessons = []
            self._next_id = 1

    def save(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "next_id": self._next_id,
                    "lessons": [asdict(ls) for ls in self._lessons],
                }, f, indent=2)
        except Exception:
            pass  # disk failure shouldn't crash the env

    # --- writing -----------------------------------------------------------

    def write(
        self,
        text: str,
        tags: List[str],
        adversary_gen: int,
        task_id: str,
        hour: int,
    ) -> Lesson:
        """Add a new lesson. Returns the stored Lesson (with id)."""
        text = text.strip()
        if not text:
            raise ValueError("Lesson text cannot be empty")
        # Deduplicate: if an almost-identical lesson exists, reinforce it
        for existing in self._lessons:
            if existing.text.lower() == text.lower():
                existing.citations += 0  # no-op but keep for clarity
                existing.last_used_ts = time.time()
                return existing

        lesson = Lesson(
            lesson_id=f"L-{self._next_id:04d}",
            text=text[:240],  # cap length
            tags=sorted(set(tags)),
            adversary_gen=adversary_gen,
            task_id=task_id,
            hour=hour,
        )
        self._next_id += 1
        self._lessons.append(lesson)
        self._prune()
        return lesson

    # --- reading -----------------------------------------------------------

    def retrieve(
        self,
        tags: Optional[List[str]] = None,
        max_items: int = 6,
    ) -> List[Lesson]:
        """
        Return the top-k lessons most relevant to the current context.
        Ranking: tag overlap (primary) × utility (secondary) × recency (tie-break).
        """
        if not self._lessons:
            return []
        query_tags = set(tags or [])

        def score(ls: Lesson) -> tuple:
            overlap = len(query_tags & set(ls.tags))
            return (overlap, ls.utility, ls.last_used_ts or ls.created_ts)

        ranked = sorted(self._lessons, key=score, reverse=True)
        return ranked[:max_items]

    def get(self, lesson_id: str) -> Optional[Lesson]:
        for ls in self._lessons:
            if ls.lesson_id == lesson_id:
                return ls
        return None

    def all(self) -> List[Lesson]:
        return list(self._lessons)

    def __len__(self) -> int:
        return len(self._lessons)

    # --- citations (commander referenced a lesson) -------------------------

    def cite(self, lesson_id: str) -> bool:
        """Commander cited this lesson. Returns True if the lesson exists."""
        ls = self.get(lesson_id)
        if ls is None:
            return False
        ls.citations += 1
        ls.last_used_ts = time.time()
        return True

    def record_outcome(self, lesson_id: str, improved: bool) -> None:
        """
        After a cited lesson's episode resolves, mark whether the outcome
        was better than baseline (improved=True) or worse.
        """
        ls = self.get(lesson_id)
        if ls is None:
            return
        if improved:
            ls.wins += 1
        else:
            ls.losses += 1

    # --- maintenance -------------------------------------------------------

    def _prune(self) -> None:
        # Drop lessons below utility floor
        self._lessons = [ls for ls in self._lessons if ls.utility >= self.min_utility]
        # If still over capacity, drop the lowest-utility lessons
        if len(self._lessons) > self.capacity:
            self._lessons.sort(key=lambda ls: ls.utility, reverse=True)
            self._lessons = self._lessons[: self.capacity]

    # --- export ------------------------------------------------------------

    def as_markdown(self) -> str:
        """Human-readable dump — useful for demos and judge review.

        Lessons are grouped by the adversary generation that spawned them
        (so the document reads as a curriculum diary), sorted by utility
        inside each group, with wins/losses/citations broken out.
        """
        if not self._lessons:
            return "_(playbook is empty)_"

        total = len(self._lessons)
        positive = sum(1 for ls in self._lessons if ls.utility > 0)
        cited = sum(1 for ls in self._lessons if ls.citations > 0)
        avg_utility = sum(ls.utility for ls in self._lessons) / total

        lines: List[str] = [
            "# Citadel Council Playbook",
            "",
            f"_{total} lessons · {positive} with positive utility · "
            f"{cited} cited at least once · avg utility {avg_utility:+.2f}_",
            "",
        ]

        by_gen: Dict[int, List[Lesson]] = {}
        for ls in self._lessons:
            by_gen.setdefault(ls.adversary_gen, []).append(ls)

        gen_names = {1: "Script Kiddie", 2: "Adaptive", 3: "Deceptive APT", 4: "Live LLM Adversary"}
        for gen in sorted(by_gen.keys()):
            lessons = sorted(by_gen[gen], key=lambda x: x.utility, reverse=True)
            n = len(lessons)
            noun = "lesson" if n == 1 else "lessons"
            lines.append(f"## Gen {gen} — {gen_names.get(gen, 'Unknown')}  ({n} {noun})")
            lines.append("")
            for ls in lessons:
                tags = " ".join(f"`{t}`" for t in ls.tags)
                provenance = f"task `{ls.task_id or 'unknown'}` · hour {ls.hour}"
                stats = (
                    f"utility {ls.utility:+.2f} · "
                    f"{ls.wins}W/{ls.losses}L · cited {ls.citations}×"
                )
                lines.append(f"- **{ls.lesson_id}** — {ls.text}")
                lines.append(f"  _{stats} · {provenance}_  ")
                lines.append(f"  {tags}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level default playbook (shared across the process)
# ---------------------------------------------------------------------------

_DEFAULT_PLAYBOOK: Optional[Playbook] = None


def get_playbook() -> Playbook:
    """Return the process-wide default playbook (lazy-initialized)."""
    global _DEFAULT_PLAYBOOK
    if _DEFAULT_PLAYBOOK is None:
        _DEFAULT_PLAYBOOK = Playbook()
    return _DEFAULT_PLAYBOOK


def reset_default_playbook(path: Optional[str] = None) -> Playbook:
    """Create a fresh default playbook (used by tests and training scripts)."""
    global _DEFAULT_PLAYBOOK
    _DEFAULT_PLAYBOOK = Playbook(path=path)
    return _DEFAULT_PLAYBOOK
