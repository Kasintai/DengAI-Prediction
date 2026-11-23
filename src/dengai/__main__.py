"""Command line entry point: `python -m dengai run`."""
import argparse
import json
import logging
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR
from .pipeline import run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dengai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="validate, backtest, tune, score the report window, write submission.csv")
    run_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="folder with the DrivenData CSVs")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where results are written")
    run_parser.add_argument("--skip-compare", action="store_true",
                            help="skip the 54-candidate comparison (the slowest step; it does not affect the submission)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    summary = run(args.raw_dir, args.output_dir, compare=not args.skip_compare)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
