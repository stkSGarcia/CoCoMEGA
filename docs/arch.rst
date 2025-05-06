Architecture Overview
=====================

Below is an overview of the code structure, showing its key modules:

::

   project
   ├── conf/
   │   └── config.yaml
   └── impl/
       ├── __main__.py
       ├── config.py
       ├── problem.py
       ├── algorithm/
       │   ├── budget.py
       │   ├── base.py
       │   ├── ccea.py
       │   ├── ga.py
       │   └── rs.py
       ├── mr/
       │   ├── mr.py
       │   └── predefined.py
       ├── scenario/
       │   ├── scenario_definition.py
       │   ├── simulation_runner.py
       │   ├── route_scenario.py
       │   ├── Interfuser_scenario_evaluator.py
       │   └── interfuser_agent.py
       └── utils/
           ├── carla_utils.py
           ├── docker_utils.py
           ├── trajectory.py
           ├── visualization.py
           ├── metrics.py
           └── math_utils.py

Core Components in ``impl/``
----------------------------

- ``__main__.py``: The entry point of the application. It defines the command-line interface (using ``python -m impl``
  invocation) for running searches, training, data collection, etc. It parses arguments (e.g., selecting the algorithm
  or mode) and invokes the appropriate functions or classes (e.g., starting a search with a specified algorithm).
- ``config.py``: Configuration management module. It loads settings from config YAML file (located in
  ``conf/config.yaml``) and makes them available via a global ``CONFIG`` object. All other parts of the code use this
  for parameters like search budget, population sizes, scenario settings, file paths, Docker image configs, etc.
- ``problem.py``: Defines the optimization problem and integrates all components. This sets up the evolutionary framework:
  it registers genetic operators, and defines the fitness function. The ``problem.py`` ties together scenarios from
  ``scenario/``, perturbations/MRs from ``mr/``, and algorithms from ``algorithm/``.
- ``algorithm/``: Contains implementations of search algorithms used to generate diverse test cases.
- ``mr/``: Defines metamorphic relations and mechanisms to evaluate them.
- ``scenario/``: Manages scenario definition, execution, and evaluation in simulation environments.
- ``utils/``: Offers utility functions supporting simulation interaction, mathematical operations, and result visualization.

Algorithm Module (``impl/algorithm/``)
--------------------------------------

The algorithm module contains implementations of the search strategies, including CoCoMEGA's custom algorithm and baseline methods:

- ``base.py``: Defines ``BaseAlgorithm``, an abstract base class that provides common functionality for all search
  algorithms (initialization, logging setup, result recording, etc.). It sets up the general solve loop structure and
  statistics gathering using *DEAP* tools.
- ``ccea.py``: Implements the Cooperative Co-Evolutionary Algorithm (CCEA) that powers CoCoMEGA. This algorithm co-evolves
  two populations, one of base scenarios and one of perturbations, collaborating to find scenario combinations that
  maximize MR violations.
- ``ga.py``: A Standard Genetic Algorithm (SGA) implementation. It evolves a single population of complete solutions
  (each solution encoding a scenario and its perturbations together) using crossover and mutation. This serves as a
  baseline to compare against CoCoMEGA's co-evolutionary approach.
- ``rs.py``: A Random Search (RS) strategy. Instead of evolving populations, this method randomly samples scenarios and
  perturbations. It's used as a simple baseline to assess the benefit of guided search.

Metamorphic Relations Module (``impl/mr/``)
-------------------------------------------

The ``impl/mr/`` module defines how metamorphic relations are represented and checked. This is central to encoding the
testing oracle for CoCoMEGA (what constitutes a failure or interesting finding in the autonomous driving context):

- ``mr.py``: Defines core classes for metamorphic relations:

    - ``Perturbation``: Represents a single modification to apply to a scenario (e.g., adding a pedestrian, changing
      weather conditions). It includes the category of the perturbation (what is being changed), the operation type
      (add, remove, replace), and the value or range of change.
    - ``Relation``: An abstract base class for the expected output relation. Subclasses like *Invariance*, *Decreasing*,
      and *Increasing* specify the type of effect we expect on a certain metric due to the perturbation (e.g., invariance
      means the metric should not change, decreasing means the metric value should go down).
    - ``MR``: Represents a full metamorphic relation, which ties together one or more ``Perturbation`` instances with an
      expected output relation. For example, an MR might consist of a perturbation "add a pedestrian in path" and an
      expected relation "ego vehicle's speed should decrease".
    - ``MRSet``: A collection of multiple MRs, which can be used together if needed (the framework can consider several
      MRs at once or switch between sets for different experiments). This module also provides methods to apply
      perturbations to a scenario and to evaluate whether an MR is violated by comparing simulation outcomes from the
      original and perturbed scenarios.

- ``predefined.py``: Contains definitions of specific metamorphic relations used in the project, corresponding to the
  descriptions from the paper/experiments. It uses the classes from ``impl/mr.py`` to build concrete instances.
  For example, it defines MR1–MR13 as described in the documentation (such as "adding a pedestrian on the roadside
  should cause the ego vehicle to slow down"). Each MR is created by specifying the needed ``PerturbationFactory`` (a
  helper to generate perturbations of a certain type) and the expected relation (``slow`` for speed decrease,
  ``steer_keep`` for steering invariance, etc.).

Scenario Module (``impl/scenario/``)
------------------------------------

The scenario module manages the generation, execution, and evaluation of driving scenarios in the CARLA simulator.
It interfaces with InterFuser agent to obtain the outcomes needed to evaluate MRs.
Major components of this module include:

- ``scenario_definition.py``: Defines the ``ScenarioDefinition`` class and related data structures that represent a
  driving scenario. A scenario includes the route (or road configuration), traffic actors (ego vehicle, other vehicles,
  pedestrians), environmental conditions (weather, brightness), and any specific parameters or positions. This file also
  provides methods to randomly generate scenarios within defined bounds (e.g., placing actors within certain regions)
  and to apply perturbations to create follow-up scenarios. It establishes how scenarios are represented as individuals
  for the search (including how to mutate them or measure distance between scenarios). Additionally, classes for
  different actor types (e.g., ``Vehicle``, ``Walker``) inherit from a base ``Actor`` class here, and a ``Boundary``
  class is used to specify allowable ranges/regions for scenario elements.
- ``scenario_manager.py``: Manages the execution lifecycle of scenarios, largely leveraging CARLA's ScenarioRunner
  infrastructure. This component takes care of initializing the CARLA world, spawning actors, and ticking the simulation.
  It controls when a scenario starts and stops, monitors timeouts, and ensures the simulation runs safely. (This file is
  adapted from CARLA's built-in scenario management code and includes logic like handling the simulation clock (GameTime)
  and watch-dog timers to recover from stalls.)
- ``simulation_runner.py``: Orchestrates the actual running of scenarios and collection of results for the search. It is
  responsible for launching simulations (potentially in separate processes or using Docker) and obtaining the outcome
  data needed to evaluate MRs. Key functionalities of simulation_runner include initializing the CARLA server, loading
  required maps, and parallelizing scenario executions.
- ``Interfuser_scenario_evaluator.py``: This sets up the scenario evaluation environment. It loads a scenario, runs the
  Interfuser agent through the scenario, and tracks scenario data for later evaluation.
- ``interfuser_agent.py``: Defines the integration for the InterFuser agent within the simulation. This includes loading
  the Interfuser model and initializing sensors. It wraps the agent so that it can take sensor data from CARLA and output
  driving controls.
