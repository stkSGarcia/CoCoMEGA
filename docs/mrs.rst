Predefined Metamorphic Relations For ADS Testing
================================================

- **MR1:** If a pedestrian appears on the roadside, then the ego-vehicle should slow down.
- **MR2:** If the driving time changes into night, then the ego-vehicle should slow down.
- **MR3**: Adding a vehicle in the front of the ego vehicle, the speed of the ego vehicle should decrease in t1% to t2%.
- **MR4**: Adding a pedestrian in the front of the ego vehicle, the speed of the ego vehicle should decrease in t1% to t2%.
- **MR5:** Changing from sunny to rainy, the speed of the ego vehicle should decrease in t1% to t2%.
- **MR6:** The density of fog (light/heavy/dense/strong/extra strong fog) in a foggy driving scene will not affect the
  steering angle of the autonomous driving systems.
- **MR7:** No matter how the driving scenes are synthesized to cope with different weather conditions (sunny and rainy),
  the driving steering angle are expected to be consistent with those under the corresponding original driving scenes.
- **MR8:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when the speed of the ego vehicle changes (increased or decreased
  by a certain factor).
- **MR9:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when adjusting (i.e., scaled down or up) the size of the ego vehicle.
- **MR10:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when adjusting (i.e., scaled down or up) the size of the target obstacle.
- **MR11:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when changing the position of the ego vehicle.
- **MR12:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when changing the speed of the target obstacle.
- **MR13:** In a given driving scenario where the ego vehicle detects a target obstacle and attempts to avoid a collision,
  the ego vehicle should keep the steering unchanged when adding additional actors.
