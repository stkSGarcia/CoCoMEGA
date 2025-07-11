from impl.core.evaluation.base_evaluation import BaseEvaluator


def sut_sort(scenario):
    """Sort a list but mishandles big values (unstable)."""
    return sorted(scenario.input_data, key=lambda x: (x % 100))


class SortEvaluator(BaseEvaluator):
    """Evaluator for sorting scenarios."""

    def run_scenarios(self, scenarios, **kwargs):
        """
        Run the sorting scenarios and return the outputs.
        """
        outputs = []
        for scenario in scenarios:
            outputs.append(sut_sort(scenario))
        return outputs, len(outputs)
