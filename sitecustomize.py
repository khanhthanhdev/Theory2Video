"""
Project-level sitecustomize to patch typing.override for Python < 3.12.

Python 3.12 adds typing.override (PEP 698). Some third-party libs import
`override` from typing, which raises ImportError on 3.11 and older.

This shim makes `typing.override` available by sourcing it from
`typing_extensions.override` when necessary. It is automatically imported
by Python at startup if this directory is on `sys.path` (default when
running commands from the project root, and also ensured for Manim
subprocesses by the renderer).
"""
try:
    import typing  # stdlib
    if not hasattr(typing, "override"):
        try:
            from typing_extensions import override as _override
            typing.override = _override  # type: ignore[attr-defined]
        except Exception:
            # typing_extensions not installed or failed; leave unmodified
            pass
except Exception:
    # Do not break interpreter startup
    pass

