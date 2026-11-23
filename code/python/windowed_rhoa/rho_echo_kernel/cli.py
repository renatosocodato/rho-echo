"""Run a reference check or simulate a serialized Rho-Echo request."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from .api import simulate
from .benchmark import main as benchmark_main
from .schemas import SimulationRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=version("rho-echo-kernel"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Replay the packaged numerical reference")
    commands.add_parser("example", help="Print a runnable example request as JSON")
    simulation = commands.add_parser("simulate", help="Simulate a request from a JSON file")
    simulation.add_argument("request", type=Path)
    args = parser.parse_args(argv)

    if args.command == "check":
        return benchmark_main(["--check"])

    if args.command == "example":
        from .benchmark import load_golden_fixture

        print(json.dumps(load_golden_fixture()["request"], indent=2))
        return 0

    try:
        request = SimulationRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8")))
        result = simulate(request)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"rho-echo: {error}\n")
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1
