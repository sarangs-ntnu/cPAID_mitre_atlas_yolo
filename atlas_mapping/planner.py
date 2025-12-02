"""Planner utilities to bind ATLAS threats and defences to perception nodes."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .models import AttackVector, DefenseMeasure, MitigationPlan, PerceptionNode


def build_mitigation_plan(
    nodes: Iterable[PerceptionNode],
    threats: Dict[str, List[AttackVector]],
    defences: Iterable[DefenseMeasure],
) -> List[MitigationPlan]:
    """Create mitigation plans for each perception node.

    Nodes are bound to their threat lists; all defences are attached so they can be
    filtered per tactic at runtime.
    """

    defence_list = list(defences)
    plans: List[MitigationPlan] = []
    for node in nodes:
        plan = MitigationPlan(node=node)
        plan.add_threats(threats.get(node.name, []))
        plan.add_defences(defence_list)
        plans.append(plan)
    return plans


def plan_summaries(plans: Iterable[MitigationPlan]) -> str:
    """Return human-readable summaries for reporting or logging."""

    return "\n\n".join(plan.summary() for plan in plans)
