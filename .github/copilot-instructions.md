Repository: hybrid_exploration — 混合式多機器人探索專案

Keep instructions short, specific, and actionable. Focus on patterns a coding agent will need to modify or extend the project.

What this repo is (big picture)
- Hybrid multi-robot exploration simulation combining centralized server task allocation and decentralized robot autonomy.
- Major components: `env.py` (environment + map IO + visualization + video buffering), `worker.py` (experiment runner), `robot.py` (robot behaviour & local graph), `server.py` (global map, utility computation, Hungarian assignment), `graph_generator.py` (node/edge generation and utilities), `graph.py` (A* and graph data structures), `sensor.py` (sensor model). See `README.md` for quick commands.

Key design & data flow notes (must-read before edits)
- Robots maintain local belief maps (`Robot.local_map`) and periodically merge with neighbors or the `Server.global_map` via `Env.merge_maps` and `Robot.update_local_awareness`.
- Graphs are stored two ways: as `Graph` objects (used inside `graph_generator`) and as lightweight dict-based `graph.edges` returned by `Graph_generator.rebuild_graph_structure` (server stores `local_map_graph` as a dict). When the server needs to plan, it converts that dict back to a `Graph` object in `Server._plan_global_path` before calling `Graph_generator.find_shortest_path`.
- Task handoff: robots can opportunistically hand off tasks during close encounters. Handoff logic and a cooldown is implemented in `robot.py` (`handoff_cooldown`, `HANDOFF_COOLDOWN` in `parameter.py`). Be careful when changing these flows.

Project-specific conventions and patterns
- Intermittent heavy vs lightweight updates: graph rebuilds are expensive and gated by `GRAPH_UPDATE_INTERVAL` (defined in `parameter.py`). Many modules follow this pattern: full rebuild every N steps, cheaper incremental updates otherwise. Preserve or mirror this pattern when adding periodic updates.
- Mixed representations: `Graph_generator` exposes both `graph.edges` (a dict of Edge objects) and `node_coords` (ndarray). Caller code often expects dicts in some places (`Server.local_map_graph`) but Graph objects in others (`graph_generator.find_shortest_path`). When converting, preserve Edge length attributes; see `Server._plan_global_path` for the conversion pattern.
- Map value semantics: maps use numeric codes: 1=obstacle, 127=unknown, 255=free. See `Env.import_map_revised` and `merge_maps`.
- Visualization + video: `Env.plot_env`, `plot_env_without_window`, `_save_frame_to_memory`, and `save_video` implement buffered vs streaming video saving. If tests or CI run headless, prefer `--no-save_video` or call `plot_env_without_window` which uses headless rendering.

Build / run / debug workflows (explicit commands)
- Setup: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
- Run a single simulation with defaults: `python worker.py`.
- Common flags used by maintainers (see `worker.py` argparse): `--TEST_MAP_INDEX`, `--TEST_AGENT_NUM`, `--plot` (interactive plotting), `--no-save_video` (do not save video), `--debug` (enable DEBUG logging).
- To run headless (CI-friendly): `python worker.py --no-save_video` or set `plot=False` when constructing `Worker`.

Integration points & external deps
- Image I/O relies on `skimage` (`skimage.io.imread`), OpenCV (`cv2`) is optional but used for streaming video if available. If cv2 is missing, the code falls back to in-memory JPEG buffers.
- `scipy.optimize.linear_sum_assignment` is used in `server.py` for assignment. Keep cost matrix shape (m x k) logic unchanged unless you update assignment rules.

Testing / safety checks for changes
- When modifying graph logic, verify both local and server planning paths: `graph_generator.find_shortest_path` and `Server._plan_global_path` must agree on node indexing and graph representation.
- When changing map values, update all places that compare to 255/127/1 (e.g., `Env.find_frontier`, `Env.import_map_revised`, `Server.calculate_coverage_ratio`).
- When editing video code, respect both streaming (_video_writer) and buffered (`frames_data`) modes. `save_video` handles both.

Small, concrete examples agents can perform
- Add a unit test that loads a sample map from `maps/easy_even/`, instantiates `Env(..., plot=False)`, runs one `Server.update_and_assign_tasks` call and asserts `coverage` in [0,1] and that no exceptions are thrown.
- If converting `graph.edges` dict -> Graph is needed elsewhere, copy the pattern from `Server._plan_global_path` (create Graph(), add nodes, iterate dict edges and add edges using Edge.length).

When to ask the human
- If a change requires modifying map semantics (different numeric codes), ask—this affects many modules.
- If you need to add non-Python dependencies (e.g., a compiled C extension or specific OpenCV build), request permission and CI instructions.

Files to open first when debugging/extension
- `README.md`, `parameter.py`, `env.py`, `worker.py`, `server.py`, `robot.py`, `graph_generator.py`, `graph.py`, `sensor.py`.

If you edit multiple modules, run a quick smoke test:
1. Create/activate venv and install requirements.
2. Run `python worker.py --TEST_MAP_INDEX 1 --TEST_AGENT_NUM 2 --no-save_video --debug` and confirm no exceptions in logs within the first 20 steps.

Please review this file and tell me which sections are unclear or need more examples; I will iterate.
