"""Shared, importable logic behind the generate.py / make_printable.py CLIs
and the web server.

Importing this package runs configure_env_and_paths() immediately, so
`import trellis_core` (before importing torch) is enough to set up the MPS
fallback env vars and sys.path that TRELLIS.2 needs. This is why it must be
the first trellis-related import in any entrypoint.
"""

from .bootstrap import configure_env_and_paths

configure_env_and_paths()
