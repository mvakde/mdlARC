from typing import Any
from task_variations.common.config import load_config as _load,PIPELINE as PROJECT_ROOT
import re
RunnerConfig=Any
PUZZLE_ID_RE=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
def load_config(path=None): return _load(path,'generate-from-summary')
