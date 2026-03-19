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
        "id": 884,
        "rule": "source_walker_focus_radius_median_scaled <= 0.10886 and source_walker_focus_speed_min_scaled > 0.07414",
        "condition": "Pedestrians ahead of the ego vehicle are generally very close, and all of them are moving, even the slowest one has a noticeable speed.",
        "diff": "On average, the updated system performs slightly worse under these conditions than the original system.",
        "explanation": "Multiple moving pedestrians at close range create competing deceleration demands. The ego vehicle must decide how much to slow down and for which pedestrian, and the unpredictability of multiple moving targets increases the likelihood of an insufficient or mistimed deceleration.",
        "coef": 0.07745424963037956,
        "support": 0.1392984239959329,
        "importance": 0.025422296847958947
    },
    {
        "id": 874,
        "rule": "follow_up_walker_focus_yaw_mean_sin <= 0.20783 and follow_up_walker_left_radius_min_scaled <= 0.18801",
        "condition": "Pedestrians ahead of the ego vehicle are mostly walking along the road direction or slightly to the right relative to the ego vehicle's heading, while a pedestrian to the left is relatively close.",
        "diff": "On average, the updated system performs moderately worse under these conditions than the original system.",
        "explanation": "The pedestrians ahead appear to be moving in a non-threatening direction, which may lead the ego vehicle to maintain its current speed. Meanwhile, the nearby left-side pedestrian may not trigger a sufficient deceleration response because the ego vehicle's attention is focused on the forward path.",
        "coef": 0.17167103378849846,
        "support": 0.4875444839857651,
        "importance": 0.08323117670619506
    },
    {
        "id": 849,
        "rule": "source_vehicle_left_speed_std_scaled <= 0.50097 and follow_up_walker_left_yaw_max_sin > -0.51541 and source_static_focus_angle_min_sin > -0.00707",
        "condition": "Vehicles to the left of the ego vehicle are traveling at fairly consistent speeds. No pedestrian to the left is walking strongly toward the left relative to the ego vehicle's heading. Static obstacles ahead are directly in front of or to the right of the ego vehicle.",
        "diff": "On average, the updated system performs moderately better under these conditions than the original system.",
        "explanation": "The orderly environment may give the ego vehicle a false sense of safety. Consistent traffic flow, non-threatening pedestrian behavior, and well-positioned obstacles may cause the ego vehicle to lower its vigilance and delay or skip a necessary deceleration.",
        "coef": -0.15040431914723712,
        "support": 0.057447890188103715,
        "importance": 0.07490372144425192
    },
    {
        "id": 852,
        "rule": "follow_up_walker_focus_angle_median_sin > -0.15199",
        "condition": "Pedestrians ahead of the ego vehicle are roughly in front of it or slightly to its right, rather than to the left.",
        "diff": "On average, the updated system performs marginally better under these conditions than the original system.",
        "explanation": "Even though pedestrians are within the ego vehicle's strong forward perception zone, the ego vehicle must correctly assess their distance and movement to determine the right deceleration response. Overconfidence in forward detection capability can still lead to insufficient deceleration.",
        "coef": -0.04490911085556727,
        "support": 0.6182003050330452,
        "importance": 0.02070031191045714
    },
    {
        "id": 847,
        "rule": "follow_up_walker_left_yaw_max_sin > 0.20871",
        "condition": "At least one pedestrian to the left of the ego vehicle is walking noticeably toward the right relative to the ego vehicle's heading.",
        "diff": "On average, the updated system performs marginally better under these conditions than the original system.",
        "explanation": "A left-side pedestrian moving rightward may be crossing toward the ego vehicle's path. The ego vehicle must determine whether the pedestrian will continue crossing or stop, and the uncertainty in predicting this behavior can lead to a delayed or insufficient deceleration.",
        "coef": -0.04409438661024402,
        "support": 0.43924758515505846,
        "importance": 0.02187494552566524
    },
    {
        "id": 877,
        "rule": "follow_up_vehicle_focus_angle_min_sin > -0.35091 and source_walker_right_yaw_std_sin > -0.15233 and follow_up_walker_focus_angle_max_cos <= 0.99336",
        "condition": "Vehicles ahead of the ego vehicle are not far to its left, they are roughly centered or to its right.",
        "diff": "On average, the updated system performs marginally worse under these conditions than the original system.",
        "explanation": "Vehicles in or near the ego vehicle's lane require the ego vehicle to balance maintaining safe following distance with the need to decelerate for other factors. The presence of a lead vehicle may discourage the ego vehicle from braking as aggressively as the situation requires.",
        "coef": 0.05336318883096931,
        "support": 0.16573462125063548,
        "importance": 0.026467890911772447
    },
    {
        "id": 3,
        "rule": "follow_up_ego_speed_scaled <= 0.04889 and follow_up_brightness <= 1.5",
        "condition": "The ego vehicle is initially traveling at a low speed under very dark lighting conditions, either deep nighttime or late twilight.",
        "diff": "On average, the updated system performs better under these conditions than the original system.",
        "coef": -0.195351485,
        "support": 0.19267921,
        "importance": 0.076282276
    },
    {
        "id": 5,
        "rule": "source_vehicle_left_speed_median_scaled <= 0.31995 and source_vehicle_focus_yaw_mean_sin > -0.95151 and source_walker_focus_yaw_mean_cos > -0.93465",
        "condition": "The vehicles to the left of the ego vehicle are not moving at a high speed, the vehicles in front of the ego vehicle are facing to the right relative to the ego vehicle's direction.",
        "diff": "On average, the updated system performs worse under these conditions than the original system.",
        "coef": 0.164368135,
        "support": 0.03457041,
        "importance": 0.073881015
    },
    {
        "id": 18,
        "rule": "follow_up_ego_speed_scaled > 0.04865 and source_static_focus_radius_min_scaled <= 0.05883",
        "condition": "The ego vehicle is moving beyond a very low speed, while an static object in front of it is extremely close to it.",
        "diff": "On average, the updated system performs very slightly worse under these conditions than the original system.",
        "coef": 0.067431264,
        "support": 0.04168785,
        "importance": 0.033525316
    },
    {
        "id": 19,
        "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled <= 0.09264",
        "condition": "The ego vehicle is nearly stopped, and the closest pedestrian to its left is extremely close to it.",
        "diff": "On average, the updated system performs slightly worse under these conditions than the original system.",
        "coef": 0.128868709,
        "support": 0.04677173,
        "importance": 0.031924072
    }
];

