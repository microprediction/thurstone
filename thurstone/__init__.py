"""thurstone: compatibility shim.

The package's implementation has been vendored into the `winning`
package (github.com/microprediction/winning) as `winning.thurstone`,
where development continues. Every public name and every submodule
import (`from thurstone.density import Density`, `import
thurstone.lattice`, ...) resolves to the vendored code, so existing
consumers work unchanged. New code should import `winning.thurstone`
directly.
"""

import importlib
import pkgutil
import sys

import winning.thurstone as _core
from winning.thurstone import *  # noqa: F401,F403

for _m in pkgutil.iter_modules(_core.__path__):
    sys.modules[f"thurstone.{_m.name}"] = importlib.import_module(
        f"winning.thurstone.{_m.name}")

__version__ = "0.2.0"
