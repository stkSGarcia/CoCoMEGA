const pagecontent = `
<p><strong>Thank you for participating in this evaluation.</strong></p>
    <p>
        The goal of this questionnaire is to assess the practical value of a set of automatically generated rules (conditions) for understanding the results of testing autonomous driving systems (ADS).
    </p>
    <p>
        These rules were derived from a large-scale simulation study comparing two versions of an ADS, for example, a baseline and an updated version.
        We automatically generated test scenarios using a strategy known as “metamorphic testing” (MT) to analyze how two ADS versions differ in terms of expected properties (metamorphic relations or MRs).
        Intuitively, with MT, a test scenario (original scenario) is modified (follow-up scenario) so that the change in ADS behavior can be determined and verified.
        The rules describe specific combinations of environmental and dynamic conditions (e.g., lighting, weather, traffic configuration, pedestrian positions) under which the two ADS versions diverge in their behavior, along with the direction (worse or better) and magnitude of that divergence.
        These rules aim to help testers analyze improvements and regressions across ADS versions.
    </p>
    
    <h3>How to read the rules</h3>
    <p>
        Each rule describes a set of driving conditions and scenarios under which the two ADS versions behave differently.
        For each rule, we provide:
    </p>
    <ul>
        <li><strong>Conditions Under Which the Two ADS Versions Differ:</strong> A plain-language description of conditions under which the two versions diverge.</li>
        <li><strong>Observed Behavioral Discrepancy:</strong> The direction and magnitude of the difference, whether the updated system performs better or worse, and by how much.</li>
        <li><strong>Occurrence Frequency:</strong> How often the described conditions appear among all detected behavioral discrepancies.</li>
        <li><strong>An Example Scenario Where This Rule Applies:</strong> Side-by-side recordings of how Version 1.0 and Version 2.0 execute the follow-up scenario under the described conditions.</li>
    </ul>
    
    <h3>Your role in this evaluation</h3>
    <p>
        Imagine you are a test engineer responsible for validating a software update to an autonomous driving system.
        You receive these automatically generated rules as a starting point, not as final test scripts, but as patterns that may highlight where the two system versions behave differently.
        We ask you to evaluate each rule from a practical perspective.
    </p>
    <p>Specifically, you will assess each rule across three dimensions:</p>
    <ul>
        <li><strong>Interpretability:</strong> How easy is it to understand the conditions described by the rule?</li>
        <li><strong>Test Prioritization:</strong> Does the rule help you narrow down which types of conditions to focus your testing efforts on?</li>
        <li><strong>Diagnostic Direction:</strong> Does the rule point you toward a specific area of the system to investigate further?</li>
    </ul>
    
    <h3>Metamorphic relations used</h3>
    <p>The test cases are based on the following MRs, which define expected behavioral adaptations of the ego
        vehicle. Each test case violates at least one of these MRs:</p>
    <ul>
        <li>🚶 If a pedestrian appears on the roadside, the ego vehicle should slow down.</li>
        <li>🌙 If driving time changes to night, the ego vehicle should slow down.</li>
        <li>🚗 If a vehicle is added in front of the ego vehicle, its speed should decrease.</li>
        <li>🚶‍♂️ If a pedestrian appears in front of the ego vehicle, its speed should decrease.</li>
        <li>🌧️ If the weather changes from sunny to rainy, the ego vehicle should decrease its speed.</li>
    </ul>
`;

const rules = [
    {
        "id": 1,
        "rule": "source_static_focus_yaw_min_sin <= 0.17505 and follow_up_traj_middle_angle_cos > 0.46432",
        "condition": "At least one static object in front of the ego vehicle is not facing significantly to the right relative to the ego vehicle's direction, while the planned trajectory runs mostly straight ahead through its halfway point, without steering far to either side.",
        "diff": "The updated system performs worse under these conditions than the original system.",
        "coef": 0.258581066,
        "support": 0.14743264,
        "importance": 0.12741214
    },
    {
        "id": 2,
        "rule": "source_ego_yaw_sin <= -0.99997 and source_walker_focus_yaw_mean_sin <= -0.17978",
        "condition": "The ego vehicle is moving directly north, and the pedestrians in front of it are generally facing slightly to the left relative to the ego vehicle's direction.",
        "diff": "The updated system performs notably better under these conditions than the original system.",
        "coef": -0.650721577,
        "support": 0.03914591,
        "importance": 0.116302774
    },
    {
        "id": 3,
        "rule": "follow_up_walker_focus_yaw_mean_sin > -0.99966 and source_static_focus_radius_max_scaled <= 0.2308",
        "condition": "The farthest static object in front of the ego vehicle is still close to it.",
        "diff": "The updated system performs considerably better under these conditions than the original system.",
        "coef": -0.500068513,
        "support": 0.37468226,
        "importance": 0.090385739
    },
    {
        "id": 4,
        "rule": "follow_up_ego_speed_scaled <= 0.04889 and follow_up_brightness <= 1.5",
        "condition": "The ego vehicle is traveling at a very low speed under very dark lighting conditions, either deep nighttime or late twilight.",
        "diff": "The updated system performs better under these conditions than the original system.",
        "coef": -0.195351485,
        "support": 0.19267921,
        "importance": 0.076282276
    },
    {
        "id": 5,
        "rule": "follow_up_vehicle_focus_angle_max_sin > -0.35114 and follow_up_vehicle_focus_radius_median_scaled > 0.20165 and follow_up_walker_focus_angle_mean_sin <= 0.00204 and source_static_focus_yaw_min_sin <= 0.99315 and source_walker_focus_yaw_mean_sin > 0.01036",
        "condition": "The vehicles in front of the ego vehicle are generally at far distance away. The pedestrians in front of the ego vehicle are generally positioned straight ahead or to its left, generally facing forward along the ego vehicle's direction.",
        "diff": "The updated system performs significantly worse under these conditions than the original system.",
        "coef": 0.637415927,
        "support": 0.00050839,
        "importance": 0.07486772
    },
    {
        "id": 6,
        "rule": "source_vehicle_left_speed_median_scaled <= 0.31995 and source_vehicle_focus_yaw_mean_sin > -0.95151 and source_walker_focus_yaw_mean_cos > -0.93465",
        "condition": "The vehicles to the left of the ego vehicle are not moving at a high speed, the vehicles in front of the ego vehicle are facing to the right relative to the ego vehicle's direction.",
        "diff": "The updated system performs worse under these conditions than the original system.",
        "coef": 0.164368135,
        "support": 0.03457041,
        "importance": 0.073881015
    },
    {
        "id": 7,
        "rule": "follow_up_walker_focus_radius_min_scaled <= 0.17387 and follow_up_vehicle_focus_yaw_min_cos > -0.99964 and source_traj_3q_radius_scaled > 0.38766 and source_walker_left_radius_median_scaled <= 0.20719",
        "condition": "The closest pedestrian in front and to the left of the ego vehicle are close to it. The planned trajectory extends at least a moderate distance.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.150872087,
        "support": 0.22267412,
        "importance": 0.071373347
    },
    {
        "id": 8,
        "rule": "source_walker_focus_speed_min_scaled <= 0.03111 and follow_up_vehicle_left_yaw_mean_sin > -0.64657 and follow_up_vehicle_focus_yaw_min_cos > -0.99948 and follow_up_brightness <= 1.5",
        "condition": "A pedestrian in front of the ego vehicle is nearly stationary, the vehicles to the left of the ego vehicle are generally not facing sharply to the left relative to the ego vehicle's direction. The lighting is very dark, either deep nighttime or late twilight.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.149535756,
        "support": 0.04067107,
        "importance": 0.069858322
    },
    {
        "id": 9,
        "rule": "follow_up_traj_3q_angle_sin <= 0.05425 and follow_up_vehicle_focus_yaw_max_sin > 0.281 and follow_up_walker_left_yaw_median_sin <= 0.99996 and follow_up_vehicle_focus_yaw_min_cos <= -0.89161",
        "condition": "The planned trajectory runs straight ahead through most of its length. At least one vehicle in front of the ego vehicle is facing moderately to the left relative to the ego vehicle's direction.",
        "diff": "The updated system performs moderately better under these conditions than the original system.",
        "coef": -0.341031025,
        "support": 0.04473818,
        "importance": 0.066853045
    },
    {
        "id": 10,
        "rule": "follow_up_vehicle_focus_angle_max_sin <= 0.09145 and follow_up_walker_left_yaw_min_cos > -0.16847",
        "condition": "The vehicles in front of the ego vehicle are all positioned nearly straight ahead or to its left, and the pedestrians to the left of the ego vehicle are not facing any rearward direction relative to the ego vehicle.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.115610047,
        "support": 0.24605999,
        "importance": 0.056373856
    },
    {
        "id": 11,
        "rule": "source_vehicle_left_yaw_min_sin <= -0.62742",
        "condition": "At least one vehicle in front of the ego vehicle is facing moderately to the left relative to the ego vehicle's direction.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.17977039,
        "support": 0.11235384,
        "importance": 0.05607107
    },
    {
        "id": 12,
        "rule": "follow_up_vehicle_focus_speed_min_scaled <= 0.80056 and follow_up_walker_focus_speed_mean_scaled <= 0.10282",
        "condition": "The vehicles in front of the ego vehicle are not all moving at high speed, and the pedestrians in front of the ego vehicle are moving very slowly or nearly stationary.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.124176161,
        "support": 0.58108795,
        "importance": 0.05292975
    },
    {
        "id": 13,
        "rule": "follow_up_walker_focus_angle_mean_sin <= -0.02592 and follow_up_walker_left_yaw_mean_cos <= -0.61443 and source_traj_3q_radius_scaled <= 0.43656 and source_vehicle_left_radius_max_scaled <= 0.22353",
        "condition": "The pedestrians in front of the ego vehicle are positioned slightly to its left, the pedestrians to the left of the ego vehicle are generally facing roughly rearward, and the vehicles to the left of the ego vehicle are close to it. The planned trajectory is relatively short.",
        "diff": "The updated system performs moderately worse under these conditions than the original system.",
        "coef": 0.344217256,
        "support": 0.00508388,
        "importance": 0.047978921
    },
    {
        "id": 14,
        "rule": "follow_up_walker_focus_angle_median_cos <= 0.97074",
        "condition": "The pedestrians in front of the ego vehicle are positioned at a slight angle, not perfectly aligned with the ego vehicle's forward direction.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.093525395,
        "support": 0.43924759,
        "importance": 0.04613248
    },
    {
        "id": 15,
        "rule": "source_walker_focus_radius_median_scaled > 0.10447",
        "condition": "The pedestrians in front of the ego vehicle are at least a short distance away, not extremely close.",
        "diff": "The updated system performs very slightly better under these conditions than the original system.",
        "coef": -0.09486336,
        "support": 0.37112354,
        "importance": 0.045542147
    },
    {
        "id": 16,
        "rule": "follow_up_walker_focus_yaw_median_sin <= 0.22232 and source_vehicle_right_yaw_max_cos > -0.99261",
        "condition": "The pedestrians in front of the ego vehicle are generally not facing to the right relative to the ego vehicle's direction.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.085922556,
        "support": 0.09964413,
        "importance": 0.041572171
    },
    {
        "id": 17,
        "rule": "follow_up_walker_focus_yaw_max_sin <= 0.21236 and source_walker_focus_yaw_mean_cos <= -0.51682 and source_walker_left_radius_min_scaled <= 0.14545",
        "condition": "None of the pedestrians in front of the ego vehicle are facing towards the left relative to the ego vehicle, while one of them in front of the ego vehicle is facing the opposite direction relative to the ego vehicle. The pedestrians to the left of the ego vehicle are very close to it.",
        "diff": "The updated system performs slightly worse under these conditions than the original system.",
        "coef": 0.157894319,
        "support": 0.01982715,
        "importance": 0.038898492
    },
    {
        "id": 18,
        "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled > 0.09264",
        "condition": "The ego vehicle is traveling at a very low speed, nearly stopped. The closest pedestrian to the left of the ego vehicle is at least a short distance away, not extremely close.",
        "diff": "The updated system performs slightly better under these conditions than the original system.",
        "coef": -0.114928211,
        "support": 0.11286223,
        "importance": 0.036289738
    },
    {
        "id": 19,
        "rule": "follow_up_walker_left_yaw_mean_sin <= 0.22708",
        "condition": "The pedestrians to the left of the ego vehicle are generally not facing to the right relative to the ego vehicle's direction.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.071716766,
        "support": 0.46568378,
        "importance": 0.034791376
    },
    {
        "id": 20,
        "rule": "follow_up_walker_focus_angle_max_sin > -0.49641 and follow_up_walker_right_yaw_median_sin > -0.89303 and source_traj_middle_angle_cos > -0.99758 and follow_up_weather_3 <= 0.5 and source_walker_left_radius_mean_scaled <= 0.19399",
        "condition": "At least one pedestrian in front of the ego vehicle is not positioned extremely  far to its left, and the pedestrians to the left of the ego vehicle are generally close to it. The weather is dry, with no significant rain or wet ground. The planned trajectory runs mostly straight ahead through its halfway point.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.082320053,
        "support": 0.2191154,
        "importance": 0.034460904
    },
    {
        "id": 21,
        "rule": "follow_up_ego_speed_scaled > 0.04865 and source_static_focus_radius_min_scaled <= 0.05883",
        "condition": "The ego vehicle is moving beyond a very low speed, while the closest static object in front of it is extremely close to it.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.067431264,
        "support": 0.04168785,
        "importance": 0.033525316
    },
    {
        "id": 22,
        "rule": "follow_up_vehicle_focus_yaw_min_sin > -0.99998 and source_traj_1q_angle_sin > -0.86979 and source_brightness > 4.5",
        "condition": "No vehicle in front of the ego vehicle is facing sharply to the left relative to the ego vehicle's direction. The planned trajectory begins without a strong leftward turn early on. The lighting is very dark, either deep nighttime or late twilight, despite originally being bright and at least moderately sunny.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.066136609,
        "support": 0.32079309,
        "importance": 0.032345564
    },
    {
        "id": 23,
        "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled <= 0.09264",
        "condition": "The ego vehicle is nearly stopped, and the closest pedestrian to its left is extremely close to it.",
        "diff": "The updated system performs slightly worse under these conditions than the original system.",
        "coef": 0.128868709,
        "support": 0.04677173,
        "importance": 0.031924072
    },
    {
        "id": 24,
        "rule": "follow_up_static_focus_yaw_max_sin <= 0.98623 and follow_up_walker_focus_yaw_max_sin <= 0.19634 and follow_up_walker_focus_yaw_max_sin > -0.99966 and follow_up_walker_left_yaw_min_cos > -0.94251 and source_walker_focus_yaw_max_cos > -0.99675",
        "condition": "No static object in front of the ego vehicle is facing sharply to the right relative to the ego vehicle's direction, the pedestrians in front of the ego vehicle are facing to the left relative to the ego vehicle's direction.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.06259293,
        "support": 0.09608541,
        "importance": 0.031295782
    },
    {
        "id": 25,
        "rule": "follow_up_walker_right_yaw_median_cos <= 0.10409 and source_walker_focus_yaw_max_cos > -0.99629 and source_walker_right_yaw_max_cos > -0.9999",
        "condition": "At least one pedestrian in front of the ego vehicle is not facing the opposite direction of the ego vehicle, and the pedestrians to the right of the ego vehicle are facing sideways or rearward.",
        "diff": "The updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.060821457,
        "support": 0.10777834,
        "importance": 0.030123171
    }
];

