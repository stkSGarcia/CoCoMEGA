Configuration Parameters Overview
=================================

The project uses a structured configuration file (``conf/config.yaml``) to manage various aspects of the testing framework.
Below is a high-level overview of the key parameters and sections in the configuration file:

General Settings
----------------

``debug``: Toggles debug mode (true or false).

Workspace Directories (``workspace``)
-------------------------------------

Paths for saving different types of results and intermediate data:

``root``: Base directory (default: ``out/``).

Subdirectories include:

- ``log``: Log files.
- ``result``: Execution results.
- ``visualization``: Generated visualizations.
- ``test_result``: Results of executed tests.

InterFuser Agent Settings (``interfuser``)
------------------------------------------

Configurations related to the InterFuser driving agent:

- ``repo``: Path to the InterFuser repository.
- ``agent_path``: Path to the agent implementation.
- ``versions``: Multiple agent configurations (v1, v2).

Data Collection (``data_collection``)
-------------------------------------

Configurations related to collecting training data for Interfuser.

Training Settings (``training``)
--------------------------------

Parameters for Interfuser model training, including: GPU allocation, training epochs, batch size, learning rate,
optimization method, town and weather configurations for training and validation data.

Docker Configurations (``docker``)
----------------------------------

Docker configuration for carla:

- ``enabled``: Enable Docker usage.
- ``image``: Built docker image name.
- ``memory``: Dedicated memory size.
- ``shared_memory``: Shared memory size.

Simulation Parameters (``simulation``)
--------------------------------------

Settings to control the simulation execution:

- ``parallel``: Parallelize simulation execution.
- ``retry_times``: Number of retries if simulation execution failed.
- ``display_agent``: Display agent interface.
- ``scenario_duration``: Duration of each scenario in seconds.
- ``measurement_interval``: Seconds between scenario data measurements.
- ``frame_rate``: Number of frames (ticks) per second.
- ``high_graphics``: Use Fidelity Mode with better visuals.
- ``autopilot``: Enable autopilot for other vehicles.
- ``disable_spectator``: Disable the simulation spectator for efficiency.
- ``instances``: Configuration of docker instances for CARLA (controls the extent of parallelization for scenario execution).

Runtime Settings (``runtime``)
------------------------------

Control the runtime environments for runtime data collection:

- ``parallel``: Parallelize runtime data collection with multiple runtime environments.
- ``agent``: Agent to use for data collection.
- ``routes``: Route types to use for data collection.
- ``towns``: Towns to use for data collection.
- ``weathers``: Weather indices to use for data collection.
- ``collection_delay``: Contains **\[lower, upper\]** in seconds.
  For each route, data collection starts after a random time between lower and upper.
- ``collection_duration``: The duration of data collection (in seconds).
- ``collection_interval``: The interval between each data collection (in seconds).
- ``collection_route_limit``: Maximum number of collected runtime data for each route (use 0 for no-limit).
- ``instances``: carla docker instances for simulating runtime environments (controls the extent of parallelization for data collection).

Search Algorithm Parameters (``search``)
----------------------------------------

Defines the budget and strategy of the search process:

- ``budget``: Specifies the search budget.

  - ``max_sim``: Maximum number of simulations.
  - ``max_time``: Maximum execution time.
  - ``max_gen``: Maximum number of generations.

- ``diff_testing``: Enable differential testing.
- ``runtime_data_as_seeds``: Use collected runtime data for scenario initialization.
- ``constraint``: Apply constraints to penalize solutions based on distance from runtime scenarios.
- ``multi_objective``: Enable multi-objective search.
- ``opt``: Configure diversity optimization and niching strategies.

  - ``niching``:

      - ``strategy``: niching strategy. Options are fitness sharing ``sharing``, fitness clearing ``clearing``,
        or not using niching ``None``.
      - ``punishment``: Punishment factor for fitness sharing.
      - ``scaling``: Scaling factor for fitness sharing.
      - ``capacity``: Number of winners in a niche when using fitness clearing.

  - ``diversity``: Enable diversity optimization.

Scenario Population (``scenario``)
----------------------------------

Scenario population parameters for CCEA:

- ``pop_size``: Scenario population size.
- ``archive_size``: Scenario archive size.
- ``leaderboard_pb``: Probability of generating leaderboard-type scenarios.
- ``init_pb``: Probabilities of generating different types of actors within a scenario.
- ``max_actors``: Maximum actors allowed for scenario initialization.
- ``dist_scaling``: The distance between two categorical values.
- ``tournament``: Tournament size for tournament selection.
- ``cxpb``: Crossover rate.
- ``mutpb``: Mutation rate.
- ``mut_add``: The probability of adding an actor when mutating.
- ``mut_del``: The probability of deleting an actor when mutating.

Perturbation Population (``perturbation``)
------------------------------------------

Perturbation population parameters for CCEA:

- ``pop_size``: Perturbation population size.
- ``archive_size``: Perturbation archive size.
- ``dist_scaling``: The distance between two categorical values.
- ``tournament``: Tournament size for tournament selection.
- ``cxpb``: Crossover rate.
- ``mutpb``: Mutation rate.

Violation Detection (``violation``)
-----------------------------------

Mechanisms for detecting violations of metamorphic relations:

- ``strategy``: Matching strategy.
- ``max_ego_distance``: Maximum distance in meters between ego-vehicle and object. Used for critical intervals.
- ``dtw``: Enable DTW matching.
- ``reevaluation``: Configuration of the re-evaluation mechanism to aggregate fitness values.

Trajectory Generation (``trajectory``)
--------------------------------------

Parameters for trajectory generation in scenarios:

- ``towns``: List of available towns for trajectory generation.
- ``initial_speed_limit``: Maximum initial speed for ego vehicle.
- ``junction_distance_limit``: Maximum distance from the closest junction to consider ``left``, ``right``, directions.
- ``predefined``: List of predefined trajectories.

Boundary (``boundary``)
-----------------------

Parameter boundaries for scenario/perturbation generation.

Blueprints (``blueprint``)
--------------------------

List of available actor blueprints and weathers to use.
