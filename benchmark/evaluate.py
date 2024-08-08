import argparse
from evalplus.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument("--samples", required=True, type=str)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--parallel", default=None, type=int)
    parser.add_argument("--i-just-wanna-run", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--noextreme", action="store_true", help="Omit extreme test inputs"
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
