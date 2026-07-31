"""Adapters: concrete implementations of ports for various storage and compute backends."""

# Extend namespace to include library's searchkernel.adapters
__path__: list[str] = __import__("pkgutil").extend_path(__path__, __name__)
