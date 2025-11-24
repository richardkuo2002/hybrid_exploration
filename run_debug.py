import logging
import sys
import numpy as np
import random
from driver import run_single_experiment

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug_detailed.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Set seed for reproducibility
np.random.seed(12345)
random.seed(67890)

# Run experiment: run_index=0, agent_num=3, map_index=1, graph_update_interval=None
print("Starting debug run...")
try:
    result = run_single_experiment((0, 3, 1, None))
    print("Run finished.")
    print(f"Result: {result}")
except Exception as e:
    logging.exception("Run failed with exception")
    print(f"Run failed: {e}")
