import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atar.env.atar_env import ATAREnv

env = ATAREnv(seed=None)
for _ in range(20):
    obs, info = env.reset()
    print(info["task_tier"])
