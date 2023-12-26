from tools.units_index import train_index
from train_log import utils
import pickle
import argparse
import os


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    units_path = os.path.join(args.data.train_path, 'units')
    exp_work_path = os.path.join(args.env.expdir, 'units_index')
    spk_str_list = os.listdir(units_path)
    print(" [INFO] The feature index is constructing.")
    for spk_str in spk_str_list:
        result = {}
        index = train_index(os.path.join(units_path, spk_str))
        result[spk_str] = index
        out_path = os.path.join(exp_work_path, f'spk{spk_str}.pkl')
        os.makedirs(exp_work_path, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
    print(" [INFO] Successfully build index")
