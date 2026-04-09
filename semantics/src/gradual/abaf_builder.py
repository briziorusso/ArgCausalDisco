from GradualABA.ABAF.Assumption import Assumption
from GradualABA.ABAF.Sentence import Sentence
from GradualABA.ABAF.Rule import Rule
from typing import Dict, Union
from src.constants import DEFAULT_WEIGHT
from src.gradual.abaf_opt import ABAFOptimised
from GradualABA.ABAF import ABAF


class ABAFBuilder:

    def __init__(self):
        Rule.reset_identifiers()
        Sentence.reset_identifiers()
        Assumption.reset_identifiers()
        self.name_to_assumption: Dict[str, Assumption] = dict()
        self.name_to_sentence = dict()
        self.rules = list()

    def add_assumption(self, assumption_name, initial_weight=DEFAULT_WEIGHT):
        assert assumption_name not in self.name_to_assumption, \
            f"Assumption {assumption_name} already exists."
        assumption = Assumption(assumption_name, initial_weight=initial_weight)
        self.name_to_assumption[assumption_name] = assumption
        self.name_to_sentence[assumption_name] = assumption

    def update_assumption_weight(self, assumption_name: str, weight: float):
        assert assumption_name in self.name_to_assumption, \
            f"Assumption {assumption_name} not found in assumptions!"
        self.name_to_assumption[assumption_name].update_weight(weight)

    def add_contrary(self, assumption_name, contrary_name):
        assert assumption_name in self.name_to_assumption, \
            f"When adding contrary, assumption {assumption_name} not found in assumptions!"
        assert contrary_name not in self.name_to_assumption, \
            f"When adding contrary, atom {contrary_name} already exists!."

        self.name_to_sentence[contrary_name] = Sentence(contrary_name)
        self.name_to_assumption[assumption_name].contrary = contrary_name

    def add_rule(self, head, body):
        if head not in self.name_to_sentence:
            self.name_to_sentence[head] = Sentence(head)
        head_object = self.name_to_sentence[head]

        body_objects = []
        for body_element in body:
            if body_element not in self.name_to_sentence:
                self.name_to_sentence[body_element] = Sentence(body_element)
            body_objects.append(self.name_to_sentence[body_element])

        rule = Rule(head=head_object, body=body_objects)
        self.rules.append(rule)

    def get_abaf(self, abaf_class: Union[ABAFOptimised, ABAF] = ABAFOptimised) -> Union[ABAFOptimised, ABAF]:
        """
        Returns the ABAF object with all assumptions, sentences, and rules.
        """
        return abaf_class(
            sentences=set(self.name_to_sentence.values()),
            assumptions=set(self.name_to_assumption.values()),
            rules=self.rules
        )
