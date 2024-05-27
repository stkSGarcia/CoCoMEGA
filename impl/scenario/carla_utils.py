import carla
def get_junction_topology(junction):
    """
    Given a junction, returns a two list of waypoints corresponding to the entry
    and exit lanes of the junction
    """
    def get_lane_key(waypoint):
        return str(waypoint.road_id) + '*' + str(waypoint.lane_id)

    def get_junction_entry_wp(entry_wp):
        while entry_wp.is_junction:
            entry_wps = entry_wp.previous(0.2)
            if len(entry_wps) == 0:
                return None
            entry_wp = entry_wps[0]
        return entry_wp

    def get_junction_exit_wp(exit_wp):
        while exit_wp.is_junction:
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                return None
            exit_wp = exit_wps[0]
        return exit_wp

    used_entry_lanes = []
    used_exit_lanes = []
    entry_wps = []
    exit_wps = []
    for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
        entry_wp = get_junction_entry_wp(entry_wp)
        if not entry_wp:
            continue
        if get_lane_key(entry_wp) not in used_entry_lanes:
            used_entry_lanes.append(get_lane_key(entry_wp))
            entry_wps.append(entry_wp)

        exit_wp = get_junction_exit_wp(exit_wp)
        if not exit_wp:
            continue
        if get_lane_key(exit_wp) not in used_exit_lanes:
            used_exit_lanes.append(get_lane_key(exit_wp))
            exit_wps.append(exit_wp)

    return entry_wps, exit_wps


def filter_junction_wp_direction(reference_wp, wp_list, direction='opposite'):
    """
    Given a list of entry / exit wps of a junction, filters them according to a specific direction,
    returning all waypoint part of lanes that are at 'direction' with respect to the reference.
    This might fail for complex junctions, as only the wp yaws is checked, not their relative positions
    """

    filtered_wps = []
    reference_yaw = reference_wp.transform.rotation.yaw
    for wp in wp_list:
        diff = (wp.transform.rotation.yaw - reference_yaw) % 360
        if diff > 330.0:
            wp_direction = 'ref'
        elif diff > 225.0:
            wp_direction = 'right'
        elif diff > 135.0:
            wp_direction = 'opposite'
        elif diff > 30.0:
            wp_direction = 'left'
        else:
            wp_direction = 'ref'

        if wp_direction == direction:
            filtered_wps.append(wp)

    return filtered_wps