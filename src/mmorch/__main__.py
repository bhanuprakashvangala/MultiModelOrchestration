"""Run the command line as a module: `python -m mmorch ...` is the same as `mmorch ...`."""

from mmorch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
