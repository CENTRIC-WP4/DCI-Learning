import argparse


def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(
        description="Randomly generate DCI message and encode by Polar codes"
    )
    parser.add_argument("--payload_length", default=70, type=int, help="Payload Length")
    parser.add_argument("--num_TTIs", default=40, type=int, help="Number of TTIs")
    parser.add_argument(
        "--encoded_length", default=128, type=int, help="Encoded DCI length"
    )
    parser.add_argument(
        "--correlated_bits_type",
        default="DecayingProduct",
        type=str,
        help="Choose the correlation type from KDependent, OneDependent or DecayingProduct",
    )
    parser.add_argument(
        "--CDL_model_type", default="A", type=str, help="CDL model type"
    )
    parser.add_argument(
        "--speed",
        default=6,
        type=float,
        help="UE moving speed",
    )
    parser.add_argument(
        "--delay_spread",
        default=300e-9,
        type=float,
        help="delay spread for CDL channel",
    )
    parser.add_argument(
        "--perfect_csi",
        action="store_true",
        help="enable perfect CSI estimation",
    )
    parser.add_argument(
        "--cyclic_prefix_length",
        default=6,
        type=int,
        help="cyclic prefix length",
    )
    parser.add_argument(
        "--num_ues",
        default=3,
        type=int,
        help="number of UEs in the cell",
    )
    parser.add_argument(
        "--num_RBs",
        default=6,
        type=int,
        help="number of resource blocks",
    )
    parser.add_argument(
        "--poisson_rate",
        default=0.5,
        type=float,
        help="average arrival for poisson distribution",
    )
    parser.add_argument(
        "--activation_ratios",
        default=0.3,
        type=float,
        help="random bits activation ratios",
    )
    parser.add_argument(
        "--correlation_factor",
        default=0.7,
        type=float,
        help="correlation factor for the random control bits",
    )
    parser.add_argument(
        "--max_buffer_size",
        default=8,
        type=int,
        help="maximum number of packets in the buffer",
    )
    parser.add_argument(
        "--ebn0_CDL",
        default=10,
        type=int,
        help="Eb/N0 value for the data channel of each UE",
    )
    parser.add_argument(
        "--esn0_control",
        default=4,
        type=int,
        help="Es/N0 for the control channel",
    )
    return parser
