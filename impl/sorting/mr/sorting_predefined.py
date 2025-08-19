from impl.core.mr.base_mr import MR, MRSet
from impl.sorting.mr.sorting_mr import InputPermutationFactory, AdditiveShiftFactory, SortedOutputEqualityRelation
from impl.sorting.scenario.sorting_scenario import ScenarioDefinition

input_size = 10

input_perm_factory = InputPermutationFactory(input_size)
additive_shift_factory = AdditiveShiftFactory()

equality_relation = SortedOutputEqualityRelation()

mr1 = MR([input_perm_factory], equality_relation)
mr2 = MR([additive_shift_factory], equality_relation)

#: Predefined MR set.
mr_set = MRSet([mr1, mr2], source_gen_func=ScenarioDefinition.generate_random)
