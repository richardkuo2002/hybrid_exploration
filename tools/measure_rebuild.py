import time
from env import Env


def measure(n_steps=100, n_agents=3, map_index=0):
    env = Env(n_agent=n_agents, k_size=10, map_index=map_index, plot=False)
    server = env.server
    rebuild_events = []
    for step in range(n_steps):
        t0 = time.time()
        done, coverage = server.update_and_assign_tasks(env.robot_list, env.real_map, env.find_frontier)
        t1 = time.time()
        elapsed = t1 - t0
        last = getattr(server, 'last_rebuild_time', None)
        if last is not None:
            rebuild_events.append((step, last))
            print(f"Step {step}: FULL_REBUILD took {last:.3f}s (total step {elapsed:.3f}s) coverage={coverage:.3f}")
        else:
            print(f"Step {step}: lightweight (total step {elapsed:.3f}s) coverage={coverage:.3f}")
        if done:
            print('Done at step', step)
            break
    if rebuild_events:
        avg = sum([e[1] for e in rebuild_events]) / len(rebuild_events)
        print('Rebuild events:', len(rebuild_events), 'avg rebuild time:', avg)
    else:
        print('No full rebuilds during measurement')


if __name__ == '__main__':
    measure(n_steps=60, n_agents=3, map_index=0)
