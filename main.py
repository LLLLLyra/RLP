import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import json
import numpy as np

from train.train_model import train, train_from_local_model


def main():
    parser = argparse.ArgumentParser(description="RL training scripts")
    parser.add_argument(
        "-c", "--config_file", nargs="?", default="config/config.json", type=str
    )
    parser.add_argument(
        "-l", "--train_from_local_model", nargs="?", default=False, type=bool
    )
    parser.add_argument("-o", "--output_model", required=True, type=str)
    parser.add_argument("-i", "--input_model", type=str)
    parser.add_argument("-s", "--init_state", nargs=3, type=float, required=True)
    parser.add_argument("-d", "--device_id", default=0, type=int)

    args = parser.parse_args()

    init_state = args.init_state
    if len(init_state) != 3:
        raise ValueError(f"invalid init state: {init_state}")

    init_state = np.array(list(map(float, init_state)))

    device_id = args.device_id

    to_train = args.train_from_local_model
    if to_train:
        if not args.input_model:
            raise ValueError(f"invalid input model: {args.input_model}")

    with open(args.config_file, "r") as f:
        config = json.load(f)

    model_out = args.output_model
    train_config = config["train_config"]
    train_config["use_multi_env"] = train_config["n_envs"] > 1
    if to_train:
        model_in = args.input_model
        train_from_local_model(
            model_in,
            model_out,
            config,
            init_state,
            device_id,
            train_config["use_multi_env"],
        )
    else:
        train(init_state, config, model_out, device_id, **train_config)


if __name__ == "__main__":
    main()
