from dataclasses import dataclass
from enum import Enum


class RelationEnum(str, Enum):
    dep = "dep"
    indep = "indep"


@dataclass
class Fact:
    relation: RelationEnum
    node1: int
    node2: int
    node_set: set
    score: float

    def __post_init__(self):
        if self.node1 > self.node2:
            self.node1, self.node2 = self.node2, self.node1


class SemanticEnum(str, Enum):
    ST = 'ST'  # Stable semantics
    PR = 'PR'  # Preferred' semantics (maximally complete)
    CO = 'CO'  # Complete semantics
