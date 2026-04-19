"""
Citadel — Server Entry Point

Exposes the two-agent Citadel environment as an OpenEnv FastAPI server.
The default /step endpoint takes a Commander action; when a CommanderProposal
is included in the step payload (action + justification), the env routes it
through the Oversight council before applying.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app

from models import IncidentAction, IncidentObservation
from environment import CitadelEnvironment

app = create_app(
    env=CitadelEnvironment,
    action_cls=IncidentAction,
    observation_cls=IncidentObservation,
    env_name="citadel",
)


@app.get("/")
def root():
    return {
        "name": "Citadel",
        "description": "Multi-Agent AI Defense Council — Commander + Oversight + Governance + Trust + Playbook",
        "themes": [
            "Theme 1 (Multi-Agent) + Fleet AI (Scalable Oversight)",
            "Theme 3.1 (Professional Tasks) + Scaler AI Labs (Multi-App Enterprise)",
            "Theme 4 (Self-Improvement) — adversary curriculum + shared playbook",
            "Theme 5 (Wild Card) — bidirectional trust dynamics",
        ],
        "endpoints": {
            "health": "GET /health",
            "schema": "GET /schema",
            "reset": "POST /reset {task_id: easy_1|medium_1|hard_1, adversary_gen?: 1|2|3}",
            "step": "POST /step {action: 0-17, target_system: 0-7, justification?: string, cited_lessons?: string[]}",
            "state": "GET /state",
            "websocket": "WS /ws",
        },
        "tasks": [
            "easy_1 (Suspicious External Activity — gen 1 default)",
            "medium_1 (Ransomware / Encryption Activity — gen 2 default)",
            "hard_1 (Anomalous Beacon / APT — gen 3 default)",
        ],
        "status": "running",
    }


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
