"""Usage: python -m llm_forecasting {develop,evaluate-final,forecast,explain-demo,report}."""

import argparse

import httpx

from .client import GenerationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("develop", "evaluate-final", "forecast", "explain-demo", "report"),
    )
    arguments = parser.parse_args()
    try:
        if arguments.stage == "explain-demo":
            from .explain import explain_demo

            explain_demo()
        elif arguments.stage == "report":
            from .reporting import build_combined_report

            build_combined_report()
        else:
            from .pipeline import develop, evaluate_final, forecast

            {
                "develop": develop,
                "evaluate-final": evaluate_final,
                "forecast": forecast,
            }[arguments.stage]()
    except (ValueError, OSError, GenerationError, httpx.HTTPError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Cancelled. Saved attempts remain available; rerun the same stage to resume.\n",
        )


if __name__ == "__main__":
    main()
