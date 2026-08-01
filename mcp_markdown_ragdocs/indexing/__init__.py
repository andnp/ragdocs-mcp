import pkgutil
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    __path__: list[str]

# Extend namespace to include library's searchkernel.indexing.
_package_path = cast(list[str], globals().get("__path__", []))
__path__ = pkgutil.extend_path(_package_path, __name__)
