"""One entry point for core tests and explicitly requested longer checks."""

import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known", action="store_true", help="also check arithmetic sources and actual square literals on CPU")
    parser.add_argument("--stress", action="store_true", help="also run the controlled 40,000-program CUDA exercise")
    parser.add_argument("--notebooks", action="store_true", help="also execute the two user notebooks in slackenv kernels")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(folder / "tests"))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    sys.path.insert(0, str(folder / "tests"))
    if args.known:
        from validate_known import validate_sources, validate_literals
        validate_sources()
        validate_literals()
    if args.stress:
        from stress_training import stress
        stress()
    if args.notebooks:
        from validate_notebooks import validate
        validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
