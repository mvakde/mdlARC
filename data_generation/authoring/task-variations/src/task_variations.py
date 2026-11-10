"""Expose the flat src tree under the task_variations import name."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

if __name__ == "__main__":
    from task_variations.cli import main
    raise SystemExit(main())

_spec = spec_from_file_location(__name__, Path(__file__).with_name("__init__.py"))
_package = module_from_spec(_spec)
sys.modules[__name__] = _package
_spec.loader.exec_module(_package)
