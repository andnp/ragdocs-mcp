"""searchkernel: hybrid search kernel (app + library namespace).

This is a namespace package that merges the library's searchkernel (from andnp-searchkernel)
with the app's app-specific modules.
"""

# This allows the app's searchkernel package to coexist with the library's searchkernel
# by treating it as an implicit namespace package.
__path__: list[str] = __import__("pkgutil").extend_path(__path__, __name__)
