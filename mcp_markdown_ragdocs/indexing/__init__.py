# Extend namespace to include library's searchkernel.indexing
__path__: list[str] = __import__("pkgutil").extend_path(__path__, __name__)
