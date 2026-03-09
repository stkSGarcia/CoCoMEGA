const rules = [
   {
      "rule": "source_static_focus_yaw_min_sin <= 0.17505 and follow_up_traj_middle_angle_cos > 0.46432",
      "condition": "A static object ahead of the ego vehicle was facing roughly straight ahead or slightly to the left, while the planned trajectory steers moderately to the side, half way.",
      "diff": "The updated system performs slightly worse under these conditions, showing a small increase in problematic behavior compared to the original system.",
      "coef": 0.258581066,
      "support": 0.415079674,
      "importance": 0.12741214
   },
   {
      "rule": "source_ego_yaw_sin <= -0.99997 and source_walker_focus_yaw_mean_sin <= -0.17978",
      "condition": "The ego vehicle was facing almost directly south, and the pedestrians ahead were, on average, facing slightly to the left relative to the ego vehicle.",
      "diff": "The updated system performs notably better under these conditions, showing a substantial reduction in problematic behavior.",
      "coef": -0.650721577,
      "support": 0.033035367,
      "importance": 0.116302774
   },
   {
      "rule": "follow_up_walker_focus_yaw_mean_sin > -0.99966 and source_static_focus_radius_max_scaled <= 0.2308",
      "condition": "The pedestrians ahead are not facing sharply to the left (i.e., their orientation is more neutral or rightward), while the farthest static object ahead was very close to the ego vehicle.",
      "diff": "The updated system performs considerably better under these conditions, showing a meaningful improvement.",
      "coef": -0.500068513,
      "support": 0.96618733,
      "importance": 0.090385739
   },
   {
      "rule": "follow_up_ego_speed_scaled <= 0.04889 and follow_up_brightness <= 1.5",
      "condition": "The ego vehicle is traveling at a very low speed, and the lighting is very dark — deep nighttime or late twilight.",
      "diff": "The updated system performs slightly better under these conditions, showing a modest improvement in near-darkness at very low speeds.",
      "coef": -0.195351485,
      "support": 0.187718616,
      "importance": 0.076282276
   },
   {
      "rule": "follow_up_vehicle_focus_angle_max_sin > -0.35114 and follow_up_vehicle_focus_radius_median_scaled > 0.20165 and follow_up_walker_focus_angle_mean_sin <= 0.00204 and source_static_focus_yaw_min_sin <= 0.99315 and source_walker_focus_yaw_mean_sin > 0.01036",
      "condition": "The vehicle farthest to the right among those ahead is positioned roughly ahead or only slightly to the left, the vehicles ahead (if any) are at least a short distance away, and the pedestrians ahead are positioned directly in front or slightly to the left of the ego vehicle. The static objects ahead (if any) were not facing sharply to the right, and the pedestrians ahead were facing very slightly to the right.",
      "diff": "The updated system performs significantly worse under these conditions, showing the largest increase in problematic behavior among these rules.",
      "coef": 0.637415927,
      "support": 0.01399145,
      "importance": 0.07486772
   },
   {
      "rule": "source_vehicle_left_speed_median_scaled <= 0.31995 and source_vehicle_focus_yaw_mean_sin > -0.95151 and source_walker_focus_yaw_mean_cos > -0.93465",
      "condition": "The vehicles to the left were moving at a slow speed, the vehicles ahead were not facing sharply to the left, and the pedestrians ahead were not facing strongly away from the ego vehicle.",
      "diff": "The updated system performs slightly worse under these conditions.",
      "coef": 0.164368135,
      "support": 0.719005052,
      "importance": 0.073881015
   },
   {
      "rule": "follow_up_walker_focus_radius_min_scaled <= 0.17387 and follow_up_vehicle_focus_yaw_min_cos > -0.99964 and source_traj_3q_radius_scaled > 0.38766 and source_walker_left_radius_median_scaled <= 0.20719",
      "condition": "The closest pedestrian ahead is very near the ego vehicle, and the vehicles ahead are not facing almost directly away. The planned trajectory extends to a moderate-to-long distance by its three-quarter point. The pedestrians to the left were very close to the ego vehicle.",
      "diff": "The updated system performs slightly better under these conditions, suggesting modest improvement when pedestrians are nearby on multiple sides.",
      "coef": -0.150872087,
      "support": 0.6618733,
      "importance": 0.071373347
   },
   {
      "rule": "source_walker_focus_speed_min_scaled <= 0.03111 and follow_up_vehicle_left_yaw_mean_sin > -0.64657 and follow_up_vehicle_focus_yaw_min_cos > -0.99948 and follow_up_brightness <= 1.5",
      "condition": "The slowest pedestrian ahead was nearly stationary. The vehicles to the left are not facing sharply to the left, the vehicles ahead are not oriented almost directly away, and the lighting is very dark — deep nighttime or late twilight.",
      "diff": "The updated system performs slightly better under these conditions, showing modest improvement in dark conditions with slow pedestrians.",
      "coef": -0.149535756,
      "support": 0.321803342,
      "importance": 0.069858322
   },
   {
      "rule": "follow_up_traj_3q_angle_sin <= 0.05425 and follow_up_vehicle_focus_yaw_max_sin > 0.281 and follow_up_walker_left_yaw_median_sin <= 0.99996 and follow_up_vehicle_focus_yaw_min_cos <= -0.89161",
      "condition": "The planned trajectory runs nearly straight ahead through its three-quarter point. At least one vehicle ahead is facing moderately to the right, the pedestrians to the left are not facing extremely far to the right, and the vehicles ahead include at least one oriented strongly away from the ego vehicle (facing almost the opposite direction).",
      "diff": "The updated system performs moderately better under these conditions, showing a notable improvement.",
      "coef": -0.341031025,
      "support": 0.040031092,
      "importance": 0.066853045
   },
   {
      "rule": "follow_up_vehicle_focus_angle_max_sin <= 0.09145 and follow_up_walker_left_yaw_min_cos > -0.16847",
      "condition": "The vehicles ahead are all positioned nearly straight ahead or only slightly to the left, and the pedestrians to the left are not facing strongly to the left.",
      "diff": "The updated system performs slightly better under these conditions.",
      "coef": -0.115610047,
      "support": 0.389428682,
      "importance": 0.056373856
   },
   {
      "rule": "source_vehicle_left_yaw_min_sin <= -0.62742",
      "condition": "At least one vehicle to the left was facing substantially to the left — roughly angled away from the ego vehicle's path.",
      "diff": "The updated system performs slightly better under these conditions.",
      "coef": -0.17977039,
      "support": 0.109211038,
      "importance": 0.05607107
   },
   {
      "rule": "follow_up_vehicle_focus_speed_min_scaled <= 0.80056 and follow_up_walker_focus_speed_mean_scaled <= 0.10282",
      "condition": "The vehicles ahead are not all moving at high speed, and the pedestrians ahead are moving very slowly or are nearly stationary.",
      "diff": "The updated system performs slightly better under these conditions, showing modest improvement when pedestrians ahead are barely moving.",
      "coef": -0.124176161,
      "support": 0.761368053,
      "importance": 0.05292975
   },
   {
      "rule": "follow_up_walker_focus_angle_mean_sin <= -0.02592 and follow_up_walker_left_yaw_mean_cos <= -0.61443 and source_traj_3q_radius_scaled <= 0.43656 and source_vehicle_left_radius_max_scaled <= 0.22353",
      "condition": "Pedestrians ahead are positioned slightly to the left of the ego vehicle, and the pedestrians to the left are facing substantially away (roughly backward). The planned trajectory is relatively short, reaching only a moderate distance by its three-quarter point. The vehicles to the left were very close to the ego vehicle.",
      "diff": "The updated system performs moderately worse under these conditions, indicating a notable increase in problematic behavior in crowded, short-trajectory scenarios.",
      "coef": 0.344217256,
      "support": 0.01982122,
      "importance": 0.047978921
   },
   {
      "rule": "follow_up_walker_focus_angle_median_cos <= 0.97074",
      "condition": "The median pedestrian ahead is positioned at a slight angle — not perfectly aligned with the ego vehicle's forward direction.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.093525395,
      "support": 0.418188885,
      "importance": 0.04613248
   },
   {
      "rule": "source_walker_focus_radius_median_scaled > 0.10447",
      "condition": "The pedestrians ahead were at least a short distance away from the ego vehicle (not extremely close).",
      "diff": "The updated system performs very slightly better under these conditions.",
      "coef": -0.09486336,
      "support": 0.360279829,
      "importance": 0.045542147
   },
   {
      "rule": "follow_up_walker_focus_yaw_median_sin <= 0.22232 and source_vehicle_right_yaw_max_cos > -0.99261",
      "condition": "The pedestrians ahead are facing roughly straight ahead or slightly to the left. The vehicles to the right were not oriented almost directly away from the ego vehicle.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.085922556,
      "support": 0.626117373,
      "importance": 0.041572171
   },
   {
      "rule": "follow_up_walker_focus_yaw_max_sin <= 0.21236 and source_walker_focus_yaw_mean_cos <= -0.51682 and source_walker_left_radius_min_scaled <= 0.14545",
      "condition": "The pedestrians ahead are facing roughly forward or slightly to the left. The pedestrians ahead were facing substantially away from the ego vehicle, and the pedestrians to the left were very close to the ego vehicle.",
      "diff": "The updated system performs slightly worse under these conditions, suggesting increased issues when pedestrians shift orientation while remaining nearby.",
      "coef": 0.157894319,
      "support": 0.06490478,
      "importance": 0.038898492
   },
   {
      "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled > 0.09264",
      "condition": "The ego vehicle was traveling at a very low speed — nearly stopped. The closest pedestrian to the left is at least a short distance away (not extremely close).",
      "diff": "The updated system performs slightly better under these conditions.",
      "coef": -0.114928211,
      "support": 0.112320249,
      "importance": 0.036289738
   },
   {
      "rule": "follow_up_walker_left_yaw_mean_sin <= 0.22708",
      "condition": "The pedestrians to the left are facing roughly forward or slightly to the left.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.071716766,
      "support": 0.621064905,
      "importance": 0.034791376
   },
   {
      "rule": "follow_up_walker_focus_angle_max_sin > -0.49641 and follow_up_walker_right_yaw_median_sin > -0.89303 and source_traj_middle_angle_cos > -0.99758 and follow_up_weather_3 <= 0.5 and source_walker_left_radius_mean_scaled <= 0.19399",
      "condition": "The pedestrians ahead are not positioned far to the left, the pedestrians to the right are not facing sharply to the left, and the weather is not wet (no significant rain or wet ground). The planned trajectory's midpoint angle indicates a fairly standard path. The pedestrians to the left were very close to the ego vehicle.",
      "diff": "The updated system performs very slightly worse under these conditions, with a small increase in problematic behavior around closely positioned pedestrians in dry conditions.",
      "coef": 0.082320053,
      "support": 0.773416246,
      "importance": 0.034460904
   },
   {
      "rule": "follow_up_ego_speed_scaled > 0.04865 and source_static_focus_radius_min_scaled <= 0.05883",
      "condition": "The ego vehicle is moving at more than a very low speed, while the closest static object ahead was extremely close to the ego vehicle.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.067431264,
      "support": 0.446949087,
      "importance": 0.033525316
   },
   {
      "rule": "follow_up_vehicle_focus_yaw_min_sin > -0.99998 and source_traj_1q_angle_sin > -0.86979 and source_brightness > 4.5",
      "condition": "The vehicles ahead are not all facing sharply to the left. The planned trajectory begins without a strong leftward turn early on. The lighting was bright — at least moderately sunny.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.066136609,
      "support": 0.396035756,
      "importance": 0.032345564
   },
   {
      "rule": "source_ego_speed_scaled <= 0.03473 and follow_up_walker_left_radius_min_scaled <= 0.09264",
      "condition": "The ego vehicle was nearly stopped. The closest pedestrian to the left is extremely close to the ego vehicle.",
      "diff": "The updated system performs slightly worse under these conditions, indicating increased problematic behavior when a very nearby pedestrian appears to the left while the ego vehicle is nearly stationary.",
      "coef": 0.128868709,
      "support": 0.065682083,
      "importance": 0.031924072
   },
   {
      "rule": "follow_up_static_focus_yaw_max_sin <= 0.98623 and follow_up_walker_focus_yaw_max_sin <= 0.19634 and follow_up_walker_focus_yaw_max_sin > -0.99966 and follow_up_walker_left_yaw_min_cos > -0.94251 and source_walker_focus_yaw_max_cos > -0.99675",
      "condition": "The static objects ahead are not facing sharply to the right, and the pedestrians ahead are facing roughly forward or slightly to the left — within a narrow forward-facing range. The pedestrians to the left are not facing strongly away from the ego vehicle. The pedestrians ahead were not oriented almost directly away.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.06259293,
      "support": 0.503303537,
      "importance": 0.031295782
   },
   {
      "rule": "follow_up_walker_right_yaw_median_cos <= 0.10409 and source_walker_focus_yaw_max_cos > -0.99629 and source_walker_right_yaw_max_cos > -0.9999",
      "condition": "The pedestrians to the right are facing mostly forward (their lateral-facing component is small). The pedestrians ahead were not facing almost directly away, and the pedestrians to the right were also not oriented almost directly away.",
      "diff": "The updated system performs very slightly worse under these conditions.",
      "coef": 0.060821457,
      "support": 0.431403031,
      "importance": 0.030123171
   }
];
