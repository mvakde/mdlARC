"""Task variations, with source folders numbered in pipeline order."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

# Python imports keep readable names while the source folders show step order.
# Installed distributions use the ordinary package names supplied by setuptools.
for _name, _folder in (
    ("summary", "01-summary"),
    ("generate", "02-generate"),
    ("review", "03-review"),
    ("improve", "04-improve"),
    ("programs", "05-programs"),
):
    _path = Path(__file__).parent / _folder
    if _path.is_dir():
        _spec = spec_from_file_location(f"{__name__}.{_name}", _path / "__init__.py")
        _module = module_from_spec(_spec)
        sys.modules[_spec.name] = _module
        _spec.loader.exec_module(_module)
        globals()[_name] = _module
