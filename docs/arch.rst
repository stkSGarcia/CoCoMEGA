Architecture Overview
=====================

Below is an overview of the code structure, showing its key modules:

.. code-block:: text

   project
   ├── conf/
   │   └── config.yaml
   ├── docs/
   └── impl/
       ├── __main__.py
       ├── config.py
       ├── problem.py
       ├── core/
       │   ├── domain_factory.py
       │   ├── algorithm/
       │   │   ├── budget.py
       │   │   ├── base.py
       │   │   ├── ccea.py
       │   │   ├── ga.py
       │   │   └── rs.py
       │   ├── evaluation/
       │   │   └── base_evaluation.py
       │   ├── mr/
       │   │   └── base_mr.py
       │   └── scenario/
       │       └── base_scenario.py
       ├── domain_template/
       │   ├── config.yaml
       │   ├── register.py
       │   ├── evaluation/
       │   │   └── domain_evaluation.py
       │   ├── mr/
       │   │   ├── domain_mr.py
       │   │   └── domain_predefined.py
       │   └── scenario/
       │       └── domain_scenario.py
       └── [domain] (using `ads` as an example)
           ├── config.yaml
           ├── register.py
           ├── evaluation/
           │   ├── scenario_manager.py
           │   ├── simulation_runner.py
           │   └── interfuser_scenario_evaluator.py
           ├── mr/
           │   ├── mr.py
           │   └── predefined.py
           ├── scenario/
           │   ├── runtime.py
           │   └── scenario_definition.py
           ├── agent/
           │   ├── interfuser_agent.py
           │   └── interfuser_config_[version].py
           └── utils/
               ├── carla_utils.py
               ├── docker_utils.py
               ├── math_utils.py
               ├── process_utils.py
               ├── trajectory.py
               ├── visualization.py
               ├── metrics.py
               └── explain.py

Core Components in ``impl/``
----------------------------

- ``__main__.py``: The entry point of the application. It defines the command-line interface (using ``python -m impl``
  invocation) for running searches, training, data collection, etc. It parses arguments (e.g., selecting the algorithm
  or mode) and invokes the appropriate functions or classes (e.g., starting a search with a specified algorithm).
- ``config.py``: Configuration management module. It loads settings from config YAML file (located in
  ``conf/config.yaml``) and makes them available via a global ``CONFIG`` object. All other parts of the code use this
  for parameters like search budget, population sizes, scenario settings, file paths, Docker image configs, etc.
- ``problem.py``: Defines the optimization problem and integrates all components. This sets up the evolutionary framework:
  it registers genetic operators, and defines the fitness function. The ``problem.py`` ties together scenarios,
  perturbations/MRs, and algorithms from the registered domain.
- ``core/``: Contains the domain-agnostic components. This module provides implementations of search algorithms and
  baseline methods, along with the base classes for scenarios and perturbations that define common functionalities. It
  also includes a domain factory that allows seamless integration of customized domains.
- ``domain_template/``: Provides templates for creating custom domains. These templates outline all the methods that need
  to be implemented to ensure compatibility and proper functionality within the framework.

Core Module (``impl/core/``)
----------------------------

The core module provides domain-agnostic components:

- ``domain_factory.py``: Loads domain-specific components by dynamically importing the domain registry module and
  retrieving its defined elements. It also manages the loading of domain-specific configuration files.
- ``algorithm/``: Contains implementations of search algorithms used to generate diverse test cases.
- ``evaluation/``: Provides a base class ``BaseEvaluator`` (in ``evaluation/base_evaluation.py``) for managing scenario
  execution and evaluation within simulation environments.
- ``scenario/``: Provides a base class ``AbstractScenarioDefinition`` (in ``scenario/base_scenario.py``) for defining
  scenarios, together with common functionalities.
- ``mr/``: Provides several base classes (in ``mr/base_mr.py``) for defining metamorphic relations and evaluation
  mechanisms, along with shared functionalities.

Algorithm Module (``impl/core/algorithm/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm module contains implementations of the search strategies, including CoCoMagic's custom algorithm and baseline methods:

- ``base.py``: Defines ``BaseAlgorithm``, an abstract base class that provides common functionality for all search
  algorithms (initialization, logging setup, result recording, etc.). It sets up the general solve loop structure and
  statistics gathering using *DEAP* tools.
- ``budget.py``: Defines the budget used in the search.
- ``ccea.py``: Implements the Cooperative Co-Evolutionary Algorithm (CCEA) that powers CoCoMagic. This algorithm co-evolves
  two populations, one of base scenarios and one of perturbations, collaborating to find scenario combinations that
  maximize the defined fitness. It also integrates constraining mechanisms and population initialization strategies.
- ``ga.py``: A Standard Genetic Algorithm (SGA) implementation. It evolves a single population of complete solutions
  (each solution encoding a scenario and its perturbations together) using crossover and mutation. This serves as a
  baseline to compare against CoCoMagic's co-evolutionary approach.
- ``rs.py``: A Random Search (RS) strategy. Instead of evolving populations, this method randomly samples scenarios and
  perturbations. It's used as a simple baseline to assess the benefit of guided search.

Metamorphic Relation Module (``impl/core/mr/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metamorphic relation module constains several abstract base classes that define how metamorphic relations are
represented and checked. This is central to encoding the testing oracle for CoCoMagic (what constitutes a failure or
interesting finding). ``base_mr.py`` defines core classes for metamorphic relations:

- ``AbstractPerturbation``: Represents a single modification to apply to a scenario (e.g., adding a pedestrian, changing
  weather conditions). It includes the category of the perturbation (what is being changed), the operation type
  (add, remove, replace), and the value or range of change.
- ``Perturbations``: Represents a sequence of perturbations and implements several genetic operators for their manipulation.
- ``AbstractPerturbationFactory``: An abstract base class for creating specific perturbations. Subclasses of this class
  define how particular perturbations are instantiated.
- ``AbstractRelation``: An abstract base class for the expected output relation. Subclasses like ``Invariance``,
  ``Decreasing``, and ``Increasing`` specify the type of effect we expect on a certain metric due to the perturbation
  (e.g., invariance means the metric should not change, decreasing means the metric value should go down).
- ``MR``: Represents a full metamorphic relation, which ties together one or more ``Perturbation`` instances with an
  expected output relation. For example, an MR might consist of a perturbation "add a pedestrian in path" and an
  expected relation "ego vehicle's speed should decrease".
- ``MRSet``: A collection of multiple MRs, with the same output relation. These MRs can be used together at once to
  generate more complex input transformations. This module also provides methods to apply
  perturbations to a scenario and to evaluate whether an MR is violated by comparing simulation outcomes from the
  original and perturbed scenarios.

Domain Template Module (``impl/domain_template/``)
--------------------------------------------------

The domain template module defines the components required to customize your own domain.
For more details, please refer to :doc:`guide`.

- ``config.yaml``: Specifies domain-specific configuration parameters.
- ``register.py``: Registers all domain-specific modules with the core framework.
- ``evaluation/``: Implements evaluation logic for assessing system behavior on generated test cases.
- ``mr/``: Defines metamorphic relations for scenario transformations and output comparisons.
- ``scenario/``: Implements test case generation logic specific to the domain.

Domain Modules (``impl/ads/`` and ``impl/sorting``)
---------------------------------------------------

The customized domain. For example, in the autonomous driving system (``ads``) domain:

- ``evaluation/``: Manages how scenarios are executed and evaluated within the CARLA simulation environment. It interfaces
  with InterFuser agent to obtain the outcomes needed to evaluate MRs.

    - ``scenario_manager.py``: Manages the execution lifecycle of scenarios, largely leveraging CARLA's ScenarioRunner
      infrastructure. This component takes care of initializing the CARLA world, spawning actors, and ticking the
      simulation. It controls when a scenario starts and stops, monitors timeouts, and ensures the simulation runs safely.
      (This file is adapted from CARLA's built-in scenario management code and includes logic like handling the simulation
      clock (GameTime) and watch-dog timers to recover from stalls.)
    - ``simulation_runner.py``: Orchestrates the actual running of scenarios and collection of results for the search.
      It is responsible for launching simulations (potentially in separate processes or using Docker) and obtaining the
      outcome data needed to evaluate MRs. Key functionalities of ``simulation_runner`` include initializing the CARLA
      server, loading required maps, and parallelizing scenario executions.
    - ``Interfuser_scenario_evaluator.py``: This sets up the scenario evaluation environment. It loads a scenario, runs
      the Interfuser agent through the scenario, and tracks scenario data for later evaluation.

- ``mr/``: Manages metamorphic relations for testing autonmous driving systems.

    - ``mr.py``: Implements metamorphic relations for the autonomous driving domain, including scenario perturbations
      and evaluation of violation extent.
    - ``predefined.py``: Contains definitions of specific metamorphic relations used in the project, corresponding to the
      descriptions from the paper/experiments. It uses the classes from ``impl/ads/mr/base_mr.py`` to build concrete
      instances. For example, it defines MR1–MR13 as described in the documentation (such as "adding a pedestrian on the
      roadside should cause the ego vehicle to slow down", see also :doc:`mrs`). Each MR is created by specifying the
      needed ``PerturbationFactory`` (a helper to generate perturbations of a certain type) and the expected relation
      (``slow`` for speed decrease, ``steer_keep`` for steering invariance, etc.).

- ``scenario/``: Manages the generation of driving scenarios in the CARLA simulator.

    - ``runtime.py``: Implements functionality to fetch and store runtime scenarios from real-world driving data.
      This module interacts with CARLA to recreate scenarios based on logged data, allowing the search to use
      representative real-world situations as seeds. It includes methods to parse logged data, reconstruct actor
      positions and behaviors, and save them in a format compatible with the scenario definition used in the search.
    - ``scenario_definition.py``: Defines the ``ScenarioDefinition`` class and related data structures that represent a
      driving scenario. A scenario includes the route (or road configuration), traffic actors (ego vehicle, other vehicles,
      pedestrians), environmental conditions (weather, brightness), and any specific parameters or positions. This file
      also provides methods to randomly generate scenarios within defined bounds (e.g., placing actors within certain
      regions) and to apply perturbations to create follow-up scenarios. It establishes how scenarios are represented as
      individuals for the search (including how to mutate them or measure distance between scenarios). Additionally,
      classes for different actor types (e.g., ``Vehicle``, ``Walker``) inherit from a base ``Actor`` class here, and a
      ``Boundary`` class is used to specify allowable ranges/regions for scenario elements.

- ``agent/``: Manages integration and configuration for InterFuser agents under test.

    - ``interfuser_agent.py``: Defines the integration for the InterFuser agent within the simulation. This includes
      loading the Interfuser model and initializing sensors. It wraps the agent so that it can take sensor data from
      CARLA and output driving controls.
    - ``interfuser_config_[version].py``: Defines configurations for different versions of the InterFuser model.

- ``utils/``: Offers utility functions supporting simulation interaction, mathematical operations, and result visualization
  and interpretation.
