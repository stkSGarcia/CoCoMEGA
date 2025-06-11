'''
    Define predefined MRs. Combine MRs that share the same output relation into an MRSet.
    Example:
        factory1 = PerturbationFactory(...)
        factory2 = PerturbationFactory(...)
        factory3 = PerturbationFactory(...)

        relation1 = Relation(...)
        relation2 = Relation(...)


        mr1 = MR([factory1], relation1)
        mr2 = MR([factory1, factory2], relation1)
        mr3 = MR([factory3], relation2)

        mr_set1 = MRSet([mr1, mr2], source_gen_func=func1)
        mr_set2 = MRSet([mr3], source_gen_func=func2)
'''
mr_set = None
