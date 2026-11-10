from task_variations.common.config import DemoSpec
from task_variations.common.config import load_config as _load
from typing import Any
RunnerConfig=Any
def load_config(path=None): return _load(path,'summarise-task')
