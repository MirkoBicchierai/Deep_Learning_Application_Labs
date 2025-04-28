import argparse
from utils import str2bool, config_loggers

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    #Dataset choice
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Possible choose stanfordnlp/sst2 or rotten_tomatoes")

    # Exercise 1.3
    parser.add_argument("--check_baseLine", type=str2bool, default=False, help="If True compute the baseline using a linear SVM (Exercise 1.3)")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()
    config_loggers(args)

if __name__ == "__main__":
    main()