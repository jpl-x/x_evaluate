import os
import subprocess
import sys

import argparse
import numpy as np
import matplotlib as mp

from envyaml import EnvYAML


def main():
    print("Python evaluation script for X-library")
    print()
    print(F"Python executable: {sys.executable}")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(F"Script located here: {script_dir}")
    print()

    parser = argparse.ArgumentParser(description='Automatic evaluation of X library according to evaluate.yaml')
    parser.add_argument('--evaluate', type=str, default="", help='location of c++ evaluate executable')
    parser.add_argument('--configuration', type=str, default="", help="YAML file specifying what to run")
    parser.add_argument('--dataset_dir', type=str, default="", help="substitutes XVIO_DATASET_DIR in yaml file")

    args = parser.parse_args()

    if len(args.evaluate) == 0:
        try:
            print("Calling catkin_find x_vio_ros evaluate")
            stream = os.popen('catkin_find x_vio_ros evaluate')
            args.evaluate = stream.read()
        finally:
            if len(args.evaluate) == 0:
                print("Error: could not find 'evaluate' executable")
                return

    if len(args.configuration) == 0:
        # default to evaluate.yaml in same folder as this script
        args.configuration = os.path.join(script_dir, "evaluate.yaml")

    print(F"Reading '{args.configuration}'")

    if len(args.dataset_dir) > 0:
        yaml_file = EnvYAML(args.configuration, XVIO_DATASET_DIR=args.dataset_dir)
    else:
        yaml_file = EnvYAML(args.configuration)

    print(F"Processing the following datasets: {str.join(', ', (d['name'] for d in yaml_file['datasets']))}")

    for dataset in yaml_file['datasets']:
        print(".")

    print()
    print(F"Using the following 'evaluate' executable: {args.evaluate}")


if __name__ == '__main__':
    main()








