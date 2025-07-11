from typing import List

from impl.core.evaluation.base_evaluation import BaseEvaluator
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class DomainEvaluator(BaseEvaluator):
    """Domain-specific evaluator for scenarios."""
    def run_scenarios(self, scenarios: List[AbstractScenarioDefinition], **kwargs):
        # TODO Implement run_scenarios
        pass
