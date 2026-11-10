from typing import Any
from task_variations.common.config import load_config as _load
RunnerConfig=Any
def load_config(path=None): return _load(path,'generate-from-reviews')
