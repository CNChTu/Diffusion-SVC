from tools.combine_tools import NaiveAndDiffModel
import argparse


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
    parser.add_argument(
        "-nmodel",
        "--naive_model",
        type=str,
        default=None,
        required=False,
        help="path to the naive model checkpoint",
    )
    parser.add_argument(
        "-votype",
        "--vocoder_type",
        type=str,
        required=False,
        default=None,
        help="If not None, will combo designated vocoder to the combo model; else nothing to do | default: None",
    )
    parser.add_argument(
        "-vopath",
        "--vocoder_path",
        type=str,
        required=False,
        default=None,
        help="If use vocoder_type, please set vocoder_path; else nothing to do | default: None",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        required=False,
        help="cpu or cuda, auto set cpu")
    parser.add_argument(
        "-exp",
        "--exp",
        type=str,
        required=True,
        help="path to the output dir",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="name for save combo model",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    combo_model = NaiveAndDiffModel(
        diff_model_path=cmd.model,
        naive_model_path=cmd.naive_model,
        vocoder_type=cmd.vocoder_type,
        vocoder_path=cmd.vocoder_path,
        device=cmd.device)
    combo_model.save_combo_model(save_path=cmd.exp, save_name=cmd.name)
