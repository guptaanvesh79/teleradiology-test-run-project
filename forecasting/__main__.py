"""Usage: python -m forecasting {develop,evaluate-final,forecast,report}."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("develop", "evaluate-final", "forecast", "report")
    )
    args = parser.parse_args()
    try:
        if args.stage == "report":
            from .reporting import build_report

            build_report()
        else:
            from .pipeline import develop, evaluate_final, forecast

            {
                "develop": develop,
                "evaluate-final": evaluate_final,
                "forecast": forecast,
            }[args.stage]()
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
