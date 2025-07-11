.. _framework_guide:

Framework Adaptation Guide
==========================

Overview
--------

This framework provides a modular and extensible architecture for developing and testing intelligent systems. It is designed to support *domain-agnostic evaluation*, with clearly separated core infrastructure and domain-specific components. The goal is to allow engineers to integrate their own domain logic (scenarios, transformations, evaluations) with minimal changes to the core framework, enabling reuse, consistency, and automation in testing processes.

Architecture
------------

Each domain is organized in a structured directory layout. The key components are:

- ``config.yaml`` – Specifies domain-specific configuration parameters.
- ``register.py`` – Registers all domain-specific modules with the core framework.
- ``evaluation/`` – Contains evaluation logic to assess system behavior on generated test cases.
- ``mr/`` – Contains metamorphic relations (MRs) for scenario transformations and output comparisons.
- ``scenario/`` – Implements test case generation logic specific to the domain.

These modules interact with the core abstractions provided by the framework, such as ``BaseScenario``, ``MetamorphicRelation``, and ``BaseEvaluation``.

Prerequisites
-------------

- Python 3.7 or higher
- Install dependencies listed in ``requirements.txt``
- Familiarity with:
  - Python object-oriented programming
  - Basic concepts of metamorphic testing

Step-by-Step Adaptation Guide
-----------------------------

1. Create Domain Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the provided template:

.. code-block:: text

    your_domain/
    ├── __init__.py
    ├── config.yaml
    ├── register.py
    ├── evaluation/
    │   └── domain_evaluation.py
    ├── mr/
    │   ├── domain_mr.py
    │   └── domain_predefined.py
    └── scenario/
        └── domain_scenario.py

2. Configure Domain Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define/Override configuration parameters in ``config.yaml``. Note that the configuration file ``config.yaml`` would be merged with the generic configurations located in ``conf/config.yaml``. Any parameters defined in the domain-specific configuration will override the generic ones.
You can also define additional parameters specific to your domain, such as scenario evaluation settings or any other domain-specific configurations.
As an example, you can override the budget for maximum number of simulations and generations for the search algorithm, and add a new parameter specific to your domain:

.. code-block:: yaml

    search:
        budget:
            max_sim: 500
            max_gen: 100

    domain_specific_param: value

3. Implement Core Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a. Scenario Definition
^^^^^^^^^^^^^^^^^^^^^^

Create a subclass of ``BaseScenario`` in ``scenario/domain_scenario.py``:

.. code-block:: python

    from impl.core.scenario import BaseScenario

    class YourScenario(BaseScenario):
        def __init__(self):
            super().__init__()
            # Initialize parameters

        def dist(self, other, **kwargs) -> float:
            # Implement distance function
            return distance_value

        def mate(self, other, cxpb, **kwargs) -> None:
            # Implement crossover operator
            return

        def mutate(self, mutpb, **kwargs) -> None:
            # Implement mutation operator
            pass

        def __eq__(self, other) -> bool:
            # TODO Implement equality criteria
            pass

        @staticmethod
        def generate_random_scenario(*args, **kwargs):
            # Implement random scenario generation logic
            return YourScenario(*args, **kwargs)

b. Metamorphic Relations
^^^^^^^^^^^^^^^^^^^^^^^^

domain_mr.py
""""""""""""""""""""""""

Create a subclass of ``AbstractPerturbation`` and ``AbstractPerturbationFactory`` for each input transformation required in your MRs, in ``mr/domain_mr.py``:

.. code-block:: python

    from impl.core.mr.base_mr import AbstractPerturbation

    class DomainPerturbation1(AbstractPerturbation):
        def dist(self, other, **kwargs) -> float:
            # Implement distance function
            return distance_value

        def mate(self, other, cxpb, **kwargs) -> None:
            # Implement crossover operator
            return

        def mutate(self, mutpb, *args, **kwargs) -> None:
            # Implement mutation operator
            pass

        def __eq__(self, other: Any) -> bool:
            # Implement equality criteria
            pass

        def apply_perturbation(self, scenario: AbstractScenarioDefinition):
            # Implement perturbation apply function
            pass

    class DomainPerturbation1Factory(AbstractPerturbationFactory):
        def spawn(self, *args, **kwargs) -> AbstractPerturbation:
            # Create and return a random instance of DomainPerturbation1
            return DomainPerturbation1(*args, **kwargs)

    class DomainPerturbation2(AbstractPerturbation):
        def dist(self, other, **kwargs) -> float:
            # Implement distance function
            return distance_value

        def mate(self, other, cxpb, **kwargs) -> None:
            # Implement crossover operator
            return

        def mutate(self, mutpb, *args, **kwargs) -> None:
            # Implement mutation operator
            pass

        def __eq__(self, other: Any) -> bool:
            # Implement equality criteria
            pass

        def apply_perturbation(self, scenario: AbstractScenarioDefinition):
            # Implement perturbation apply function
            pass

    class DomainPerturbation2Factory(AbstractPerturbationFactory):
        def spawn(self, *args, **kwargs) -> AbstractPerturbation:
            # Create and return a random instance of DomainPerturbation2
            return DomainPerturbation2(*args, **kwargs)

Also define subclasses of ``AbstractRelation`` to define output relations for your MRs, in ``mr/domain_mr.py``:

.. code-block:: python

    from impl.core.mr.base_mr import AbstractRelation

    class DomainOutputRelation1(AbstractRelation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Initialize parameters

        is_violated(self, original_output: Any, perturbed_output: Any, **kwargs) -> (bool, float):
            # Implement violation checking logic
            return True, 1.0  # the first value indicates if the relation is violated, the second indicates the extend of violation.

    class DomainOutputRelation2(AbstractRelation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Initialize parameters

        is_violated(self, original_output: Any, perturbed_output: Any, **kwargs) -> (bool, float):
            # Implement violation checking logic
            return True, 1.0  # the first value indicates if the relation is violated, the second indicates the extend of violation.



domain_predefined.py
""""""""""""""""""""""""
Define the metamorphic relations using the perturbations and output relations defined above in ``mr/domain_predefined.py``:

.. code-block:: python

    from impl.core.mr.base_mr import MR, MRSet
    from impl.sorting.mr.domain_mr import Perturbation1Factory, Perturbation2Factory, Perturbation1, Perturbation2, \
        DomainOutputRelation1, DomainOutputRelation2
    from impl.your_domain.scenario.domain_scenario import YourScenario

    factory1 = Perturbation1Factory(...)
    factory2 = Perturbation2Factory(...)

    relation1 = DomainOutputRelation1(...)
    relation2 = DomainOutputRelation2(...)

    mr1 = MR([factory1], relation1)
    mr2 = MR([factory2], relation2)

    mr_set = MRSet([mr1, mr2], source_gen_func=YourScenario.generate_random_scenario)

c. Evaluation Logic
^^^^^^^^^^^^^^^^^^^

Create a subclass of ``BaseEvaluation`` in ``evaluation/domain_evaluation.py``:

.. code-block:: python

    from typing import List
    from impl.core.evaluation.base_evaluation import BaseEvaluator
    from impl.core.scenario.base_scenario import AbstractScenarioDefinition


    class DomainEvaluator(BaseEvaluator):
        def run_scenarios(self, scenarios: List[AbstractScenarioDefinition], **kwargs):
            # Implement scenario evaluation logic
            pass

d. Register Components
^^^^^^^^^^^^^^^^^^^^^^

You must also register your new components in ``register.py`` so they can be used by the framework. The modules to be registered include:
- Scenario: The scenario definition class.
- ScenarioInit: A function to initialize the scenario population.
- Perturbation: The perturbation factories and their definitions.
- Evaluation: The evaluation class that will assess the scenarios.
- MRSet: The set of metamorphic relations, including the perturbations and output relations.
- Config: The configuration file for the domain.

These components should be registered in the `DOMAIN_REGISTRY` dictionary in `register.py`. Here is an example of how to do this:

.. code-block:: python

    from impl.your_domain.scenario.domain_scenario import YourScenario
    from impl.your_domain.mr.domain_mr import DomainPerturbation1Factory, DomainPerturbation2Factory
    from impl.your_domain.evaluation.domain_evaluation import DomainEvaluator

    from deap import creator
    import impl.config as cfg
    from impl.core.mr.base_mr import Perturbations
    from impl.your_domain.mr.domain_predefined import mr_set
    from impl.your_domain.scenario.domain_scenario import ScenarioDefinition
    from impl.your_domain.evaluation.domain_evaluation import DomainEvaluator

    mr_set = mr_set

    def _pop_scenario():
        """Initialize the scenario population"""
        pop_scenario = [creator.Scenario(instance=mr_set.generate_scenario()) for _ in range(cfg.CONFIG["scenario"]["pop_size"])]
        return pop_scenario


    DOMAIN_REGISTRY = {
        "Scenario": ScenarioDefinition,
        "ScenarioInit": _pop_scenario,
        "Perturbation": Perturbations,
        "Evaluation": DomainEvaluator,
        "MRSet": mr_set,
        "Config": "config.yaml",
    }
e. Plug-in the Domain
^^^^^^^^^^^^^^^^^^^^^^

Finally, plug your domain into the core framework by adding it to the entry point ``problem.py``:

.. code-block:: python

    # Define domain
    domain = DomainFactory("your_domain")

Testing Your Implementation
--------------------------

After implementing all components, execute the framework with your domain:

.. code-block:: bash

    python -m impl search

Common Issues and Troubleshooting
--------------------------------

- Ensure all abstract methods are implemented in your classes
- Verify that your configuration file follows YAML syntax
- Check that all required dependencies are installed
- Validate that your domain components are properly registered

Additional Resources
------------------

- See `sorting/` directory for a reference implementation of a simple sorting domain.
- Refer to `domain_template/` for a template to start your own domain.
- Check the project's issue tracker for known problems and solutions.

Support
-------

For questions and support:

- Open an issue on the project repository
- Check existing documentation in the `docs/` directory
- Contact the maintainers via the project's communication channels
