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
    <p>Specifically, you will assess each rule across four dimensions:</p>
    <ul>
        <li><strong>Interpretability:</strong> How easy is it to understand the conditions described by the rule?</li>
        <li><strong>Test Prioritization:</strong> Does the rule help you narrow down which types of conditions to focus your testing efforts on?</li>
        <li><strong>Diagnostic Direction:</strong> Does the rule point you toward a specific area of the system to investigate further?</li>
        <li><strong>Non-Obviousness:</strong> Does the rule reveal a condition that you would not have thought to test on your own?</li>
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
        "condition": "Static objects in front of the ego vehicle are facing roughly the same direction as the ego vehicle, or slightly to its left, while the planned trajectory runs mostly straight ahead through its halfway point, without steering far to either side.",
        "diff": "The updated system performs worse under these conditions than the original system.",
        "coef": 0.258581066,
        "support": 0.415079674,
        "importance": 0.12741214
    },
    {
        "id": 2,
        "rule": "source_ego_yaw_sin <= -0.99997 and source_walker_focus_yaw_mean_sin <= -0.17978",
        "condition": "The ego vehicle is facing almost directly north, and the pedestrians in front of it are generally facing slightly to the left of the ego vehicle's direction.",
        "diff": "The updated system performs notably better under these conditions than the original system.",
        "coef": -0.650721577,
        "support": 0.033035367,
        "importance": 0.116302774
    },
    {
        "id": 3,
        "rule": "follow_up_walker_focus_yaw_mean_sin > -0.99966 and source_static_focus_radius_max_scaled <= 0.2308",
        "condition": "The pedestrians in front of the ego vehicle are not facing sharply to its left, while the farthest static object in front of the ego vehicle is very close to it.",
        "diff": "The updated system performs considerably better under these conditions than the original system.",
        "coef": -0.500068513,
        "support": 0.96618733,
        "importance": 0.090385739
    },
    {
        "id": 4,
        "rule": "follow_up_ego_speed_scaled <= 0.04889 and follow_up_brightness <= 1.5",
        "condition": "The ego vehicle is traveling at a very low speed, and the lighting is very dark — deep nighttime or late twilight.",
        "diff": "The updated system performs slightly better under these conditions.",
        "coef": -0.195351485,
        "support": 0.187718616,
        "importance": 0.076282276
    },
    {
        "id": 5,
        "rule": "follow_up_vehicle_focus_angle_max_sin > -0.35114 and follow_up_vehicle_focus_radius_median_scaled > 0.20165 and follow_up_walker_focus_angle_mean_sin <= 0.00204 and source_static_focus_yaw_min_sin <= 0.99315 and source_walker_focus_yaw_mean_sin > 0.01036",
        "condition": "[TODO] The vehicles in the field of view of the ego vehicle are not close to the ego vehicle and they are not extremely far to the left. The pedestrians in the field of view of the ego vehicle is not positioned to the right of the ego vehicle and they are, on average, facing .",
        "diff": "The updated system performs significantly worse under these conditions, showing the largest increase in problematic behavior among these rules.",
        "coef": 0.637415927,
        "support": 0.01399145,
        "importance": 0.07486772
    },
    {
        "id": 6,
        "rule": "source_vehicle_left_speed_median_scaled <= 0.31995 and source_vehicle_focus_yaw_mean_sin > -0.95151 and source_walker_focus_yaw_mean_cos > -0.93465",
        "condition": "The vehicles to the left of the ego vehicle are moving at a slow speed, the vehicles ahead are not facing sharply to the left, and the pedestrians ahead are not facing strongly away from the ego vehicle.",
        "diff": "The updated system performs slightly worse under these conditions.",
        "coef": 0.164368135,
        "support": 0.719005052,
        "importance": 0.073881015
    },
    {
        "id": 7,
        "rule": "follow_up_walker_focus_radius_min_scaled <= 0.17387 and follow_up_vehicle_focus_yaw_min_cos > -0.99964 and source_traj_3q_radius_scaled > 0.38766 and source_walker_left_radius_median_scaled <= 0.20719",
        "condition": "The closest pedestrian ahead is very near the ego vehicle, and the vehicles ahead are not facing to the ego vehicle. The pedestrians to the left of the ego vehicle are very close to the ego vehicle.",
        "diff": "The updated system performs slightly better under these conditions, suggesting modest improvement when pedestrians are nearby on multiple sides.",
        "coef": -0.150872087,
        "support": 0.6618733,
        "importance": 0.071373347
    },
    {
        "id": 8,
        "rule": "source_walker_focus_speed_min_scaled <= 0.03111 and follow_up_vehicle_left_yaw_mean_sin > -0.64657 and follow_up_vehicle_focus_yaw_min_cos > -0.99948 and follow_up_brightness <= 1.5",
        "condition": "The slowest pedestrian ahead was nearly stationary. The vehicles to the left of the ego vehicle are not facing sharply to the left, the vehicles ahead are not facing to the ego vehicle, and the lighting is very dark — deep nighttime or late twilight.",
        "diff": "The updated system performs slightly better under these conditions, showing modest improvement in dark conditions with slow pedestrians.",
        "coef": -0.149535756,
        "support": 0.321803342,
        "importance": 0.069858322
    },
    {
        "id": 9,
        "rule": "follow_up_traj_3q_angle_sin <= 0.05425 and follow_up_vehicle_focus_yaw_max_sin > 0.281 and follow_up_walker_left_yaw_median_sin <= 0.99996 and follow_up_vehicle_focus_yaw_min_cos <= -0.89161",
        "condition": "The planned trajectory runs nearly straight ahead through its three-quarter point. At least one vehicle ahead is facing moderately to the right, the pedestrians to the left are not facing extremely far to the right, and the vehicles ahead include at least one oriented strongly away from the ego vehicle (facing almost the opposite direction).",
        "diff": "The updated system performs moderately better under these conditions, showing a notable improvement.",
        "coef": -0.341031025,
        "support": 0.040031092,
        "importance": 0.066853045
    },
    {
        "id": 10,
        "rule": "follow_up_vehicle_focus_angle_max_sin <= 0.09145 and follow_up_walker_left_yaw_min_cos > -0.16847",
        "condition": "The vehicles ahead are all positioned nearly straight ahead or only slightly to the left, and the pedestrians to the left are not facing strongly to the left.",
        "diff": "The updated system performs slightly better under these conditions.",
        "coef": -0.115610047,
        "support": 0.389428682,
        "importance": 0.056373856
    },
    {
        "id": 11,
        "rule": "source_vehicle_left_yaw_min_sin <= -0.62742",
        "condition": "At least one vehicle to the left was facing substantially to the left — roughly angled away from the ego vehicle's path.",
        "diff": "The updated system performs slightly better under these conditions.",
        "coef": -0.17977039,
        "support": 0.109211038,
        "importance": 0.05607107
    },
    {
        "id": 12,
        "rule": "follow_up_vehicle_focus_speed_min_scaled <= 0.80056 and follow_up_walker_focus_speed_mean_scaled <= 0.10282",
        "condition": "The vehicles ahead are not all moving at high speed, and the pedestrians ahead are moving very slowly or are nearly stationary.",
        "diff": "The updated system performs slightly better under these conditions, showing modest improvement when pedestrians ahead are barely moving.",
        "coef": -0.124176161,
        "support": 0.761368053,
        "importance": 0.05292975
    },
    {
        "id": 13,
        "rule": "follow_up_walker_focus_angle_mean_sin <= -0.02592 and follow_up_walker_left_yaw_mean_cos <= -0.61443 and source_traj_3q_radius_scaled <= 0.43656 and source_vehicle_left_radius_max_scaled <= 0.22353",
        "condition": "Pedestrians ahead are positioned slightly to the left of the ego vehicle, and the pedestrians to the left are facing substantially away (roughly backward). The planned trajectory is relatively short, reaching only a moderate distance by its three-quarter point. The vehicles to the left were very close to the ego vehicle.",
        "diff": "The updated system performs moderately worse under these conditions, indicating a notable increase in problematic behavior in crowded, short-trajectory scenarios.",
        "coef": 0.344217256,
        "support": 0.01982122,
        "importance": 0.047978921
    },
    {
        "id": 14,
        "rule": "follow_up_walker_focus_angle_median_cos <= 0.97074",
        "condition": "The median pedestrian ahead is positioned at a slight angle — not perfectly aligned with the ego vehicle's forward direction.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.093525395,
        "support": 0.418188885,
        "importance": 0.04613248
    },
    {
        "id": 15,
        "rule": "source_walker_focus_radius_median_scaled > 0.10447",
        "condition": "The pedestrians ahead were at least a short distance away from the ego vehicle (not extremely close).",
        "diff": "The updated system performs very slightly better under these conditions.",
        "coef": -0.09486336,
        "support": 0.360279829,
        "importance": 0.045542147
    },
    {
        "id": 16,
        "rule": "follow_up_walker_focus_yaw_median_sin <= 0.22232 and source_vehicle_right_yaw_max_cos > -0.99261",
        "condition": "The pedestrians ahead are facing roughly straight ahead or slightly to the left. The vehicles to the right were not oriented almost directly away from the ego vehicle.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.085922556,
        "support": 0.626117373,
        "importance": 0.041572171
    },
    {
        "id": 17,
        "rule": "follow_up_walker_focus_yaw_max_sin <= 0.21236 and source_walker_focus_yaw_mean_cos <= -0.51682 and source_walker_left_radius_min_scaled <= 0.14545",
        "condition": "The pedestrians ahead are facing roughly forward or slightly to the left. The pedestrians ahead were facing substantially away from the ego vehicle, and the pedestrians to the left were very close to the ego vehicle.",
        "diff": "The updated system performs slightly worse under these conditions, suggesting increased issues when pedestrians shift orientation while remaining nearby.",
        "coef": 0.157894319,
        "support": 0.06490478,
        "importance": 0.038898492
    },
    {
        "id": 18,
        "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled > 0.09264",
        "condition": "The ego vehicle was traveling at a very low speed — nearly stopped. The closest pedestrian to the left is at least a short distance away (not extremely close).",
        "diff": "The updated system performs slightly better under these conditions.",
        "coef": -0.114928211,
        "support": 0.112320249,
        "importance": 0.036289738
    },
    {
        "id": 19,
        "rule": "follow_up_walker_left_yaw_mean_sin <= 0.22708",
        "condition": "The pedestrians to the left are facing roughly forward or slightly to the left.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.071716766,
        "support": 0.621064905,
        "importance": 0.034791376
    },
    {
        "id": 20,
        "rule": "follow_up_walker_focus_angle_max_sin > -0.49641 and follow_up_walker_right_yaw_median_sin > -0.89303 and source_traj_middle_angle_cos > -0.99758 and follow_up_weather_3 <= 0.5 and source_walker_left_radius_mean_scaled <= 0.19399",
        "condition": "The pedestrians ahead are not positioned far to the left, the pedestrians to the right are not facing sharply to the left, and the weather is not wet (no significant rain or wet ground). The planned trajectory's midpoint angle indicates a fairly standard path. The pedestrians to the left were very close to the ego vehicle.",
        "diff": "The updated system performs very slightly worse under these conditions, with a small increase in problematic behavior around closely positioned pedestrians in dry conditions.",
        "coef": 0.082320053,
        "support": 0.773416246,
        "importance": 0.034460904
    },
    {
        "id": 21,
        "rule": "follow_up_ego_speed_scaled > 0.04865 and source_static_focus_radius_min_scaled <= 0.05883",
        "condition": "The ego vehicle is moving at more than a very low speed, while the closest static object ahead was extremely close to the ego vehicle.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.067431264,
        "support": 0.446949087,
        "importance": 0.033525316
    },
    {
        "id": 22,
        "rule": "follow_up_vehicle_focus_yaw_min_sin > -0.99998 and source_traj_1q_angle_sin > -0.86979 and source_brightness > 4.5",
        "condition": "The vehicles ahead are not all facing sharply to the left. The planned trajectory begins without a strong leftward turn early on. The lighting was bright — at least moderately sunny.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.066136609,
        "support": 0.396035756,
        "importance": 0.032345564
    },
    {
        "id": 23,
        "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled <= 0.09264",
        "condition": "The ego vehicle was nearly stopped. The closest pedestrian to the left is extremely close to the ego vehicle.",
        "diff": "The updated system performs slightly worse under these conditions, indicating increased problematic behavior when a very nearby pedestrian appears to the left while the ego vehicle is nearly stationary.",
        "coef": 0.128868709,
        "support": 0.065682083,
        "importance": 0.031924072
    },
    {
        "id": 24,
        "rule": "follow_up_static_focus_yaw_max_sin <= 0.98623 and follow_up_walker_focus_yaw_max_sin <= 0.19634 and follow_up_walker_focus_yaw_max_sin > -0.99966 and follow_up_walker_left_yaw_min_cos > -0.94251 and source_walker_focus_yaw_max_cos > -0.99675",
        "condition": "The static objects ahead are not facing sharply to the right, and the pedestrians ahead are facing roughly forward or slightly to the left — within a narrow forward-facing range. The pedestrians to the left are not facing strongly away from the ego vehicle. The pedestrians ahead were not oriented almost directly away.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.06259293,
        "support": 0.503303537,
        "importance": 0.031295782
    },
    {
        "id": 25,
        "rule": "follow_up_walker_right_yaw_median_cos <= 0.10409 and source_walker_focus_yaw_max_cos > -0.99629 and source_walker_right_yaw_max_cos > -0.9999",
        "condition": "The pedestrians to the right are facing mostly forward (their lateral-facing component is small). The pedestrians ahead were not facing almost directly away, and the pedestrians to the right were also not oriented almost directly away.",
        "diff": "The updated system performs very slightly worse under these conditions.",
        "coef": 0.060821457,
        "support": 0.431403031,
        "importance": 0.030123171
    }
];

