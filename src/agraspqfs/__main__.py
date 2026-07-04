"""Allow running with `python -m agraspqfs`."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
