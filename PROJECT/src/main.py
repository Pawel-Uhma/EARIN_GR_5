import argparse
from train import train
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Face Age Classification")
    parser.add_argument("--mode", type=str, choices=["train","eval"], default="train")
    parser.add_argument("--model_path", type=str, help="Path to the saved model for evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train()
    else:
        if not args.model_path:
            raise ValueError("Please provide --model_path for evaluation mode.")
        evaluate(args.model_path)
