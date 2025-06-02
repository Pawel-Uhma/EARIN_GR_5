import argparse
from train import train
from evaluate import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Age Regression Training/Evaluation')
    parser.add_argument('--mode', choices=['train','eval'], default='train')
    parser.add_argument('--model_path', type=str, help='Path to model for evaluation')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        evaluate(model_path=args.model_path)