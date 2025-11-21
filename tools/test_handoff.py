import copy
import os
import sys

import numpy as np

# Ensure repository root is on sys.path so local modules can be imported when running via conda
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from robot import Robot


def make_robot(id, pos):
    r = Robot(
        start_position=np.array(pos), real_map_size=(20, 20), resolution=1, k_size=5
    )
    r.robot_id = id
    return r


def test_deep_copy_handoff():
    r0 = make_robot(0, [2, 2])
    r1 = make_robot(1, [2, 3])

    # r1 has a server-given target and planned_path
    r1.target_gived_by_server = True
    r1.target_pos = np.array([10, 10])
    r1.planned_path = [np.array([3, 3]), np.array([4, 4]), np.array([10, 10])]

    # r0 is returning and meets r1
    r0.is_returning = True

    # simulate communication range
    all_robots = [r0, r1]

    # call update_local_awareness but stub out sensor/merge/find_frontier
    def noop_sensor(pos, sr, local_map, real_map):
        return local_map

    def noop_merge(maps):
        return maps[0]

    def noop_frontier(dm):
        return []

    # Create a fake server with minimal attributes
    from parameter import PIXEL_UNKNOWN

    class S:
        def __init__(self):
            self.position = np.array([0, 0])
            self.global_map = np.ones((20, 20)) * PIXEL_UNKNOWN
            self.all_robot_position = [None, None]
            self.robot_in_range = [False, False]

    server = S()

    # prepare a simple real_map to satisfy sensor_work's expectations
    from parameter import PIXEL_UNKNOWN

    real_map = np.ones((20, 20), dtype=np.uint8) * PIXEL_UNKNOWN
    # run awareness: should trigger handoff (r0 takes r1's task)
    r0.update_local_awareness(
        real_map=real_map,
        all_robots=all_robots,
        server=server,
        find_frontier_func=noop_frontier,
        merge_maps_func=noop_merge,
    )

    # After handoff, ensure deep copies: mutate r1.planned_path and check r0 unaffected
    r1.planned_path.append(np.array([99, 99]))

    assert (
        len(r0.planned_path) == 3
    ), f"r0 planned_path should be length 3, got {len(r0.planned_path)}"
    assert not any(
        (np.array_equal(p, np.array([99, 99])) for p in r0.planned_path)
    ), "r0 planned_path was affected by r1 modification (shared reference)"
    assert (
        r0.handoff_cooldown == 3
    ), "r0 handoff_cooldown should be set to HANDOFF_COOLDOWN"
    assert (
        r1.handoff_cooldown == 3
    ), "r1 handoff_cooldown should be set to HANDOFF_COOLDOWN"

    print("TEST PASSED: handoff uses deep copy and cooldown set")


if __name__ == "__main__":
    test_deep_copy_handoff()
