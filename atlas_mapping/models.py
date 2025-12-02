"""Core data structures for mapping MITRE ATLAS tactics to the perception stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ATLASTactic:
    """Represents a MITRE ATLAS technique relevant to vision systems."""

    identifier: str
    name: str
    description: str

    def as_label(self) -> str:
        """Return a short label combining identifier and name."""

        return f"{self.identifier} – {self.name}"


@dataclass(frozen=True)
class AttackVector:
    """Specific attack instance linked to an ATLAS tactic."""

    tactic: ATLASTactic
    goal: str
    entry_points: Sequence[str]

    def summary(self) -> str:
        return f"{self.tactic.as_label()}: {self.goal}"


@dataclass(frozen=True)
class DefenseMeasure:
    """Mitigation or detection approach for a given attack vector."""

    title: str
    description: str
    linked_tactics: Sequence[ATLASTactic]

    def addresses(self, tactic: ATLASTactic) -> bool:
        return tactic in self.linked_tactics


@dataclass(frozen=True)
class PerceptionNode:
    """Logical component of the perception stack."""

    name: str
    responsibilities: Sequence[str]


@dataclass
class MitigationPlan:
    """Pairing between perception nodes, attack vectors, and defences."""

    node: PerceptionNode
    threats: List[AttackVector] = field(default_factory=list)
    defences: List[DefenseMeasure] = field(default_factory=list)

    def add_threats(self, new_threats: Iterable[AttackVector]) -> None:
        self.threats.extend(new_threats)

    def add_defences(self, new_defences: Iterable[DefenseMeasure]) -> None:
        self.defences.extend(new_defences)

    def matched_defences(self, tactic: ATLASTactic) -> List[DefenseMeasure]:
        """Return defences that explicitly address the given tactic."""

        return [defence for defence in self.defences if defence.addresses(tactic)]

    def summary(self) -> str:
        threat_lines = [threat.summary() for threat in self.threats]
        defence_titles = [defence.title for defence in self.defences]
        return (
            f"Perception node: {self.node.name}\n"
            f"Responsibilities: {', '.join(self.node.responsibilities)}\n"
            f"Threats: {', '.join(threat_lines)}\n"
            f"Defences: {', '.join(defence_titles)}"
        )
