from pdcch.parameters import ParametersPDCCH
from pdcch.encoder import PDCCHEncoder
from pdcch.decoder import PDCCHDecoder
from sionna.fec.polar import PolarSCLDecoder
from control_bits.system import UserEquipment, BTS, EvenResourcesNonContiguous
from control_bits.channel_model import CDLModel
import utils
import sys
import os
from baselines import (
    lossless_compression,
    train_test_split,
    CompressionStatisticCollector,
)

if __name__ == "__main__":
    ## initialize all the system parameters
    args = utils.options().parse_args()
    folder_name = "_".join(sys.argv[1:]).replace("-", "")
    folder_name = "default" if folder_name == "" else folder_name
    saver_directory = f"./output/{folder_name}"
    os.makedirs(saver_directory, exist_ok=True)

    ## initialize the channel model that each UE has.
    channel_model = CDLModel(
        args.CDL_model_type,
        args.delay_spread,
        args.perfect_csi,
        args.speed,
        args.cyclic_prefix_length,
        args.num_TTIs,
        args.num_RBs,
    )
    UEs = [
        UserEquipment(
            channel_model, args.poisson_rate, args.max_buffer_size, args.ebn0_CDL, i
        )
        for i in range(args.num_ues)
    ]
    pdcch_parameters = ParametersPDCCH(E=args.encoded_length, A=args.payload_length)
    pdcch_encoder = PDCCHEncoder(pdcch_parameters)
    pdcch_decoder = PDCCHDecoder(pdcch_parameters)
    pdcch_decoder.channel_decoder = PolarSCLDecoder(
        frozen_pos=pdcch_parameters.frozen_bit_positions, n=pdcch_parameters.N
    )
    FD_scheduler = EvenResourcesNonContiguous(args.num_RBs)

    bts = BTS(
        UEs,
        pdcch_encoder,
        pdcch_decoder,
        args.payload_length,
        args.num_RBs,
        args.num_TTIs,
        FD_scheduler,
        args.correlated_bits_type,
        args.activation_ratios,
        args.correlation_factor,
        args.esn0_control,
    )
    DCI_UE, encoded_message_UE = bts.generate_new_epoch()
    source_data = [train_test_split(x) for x in DCI_UE.values()]

    compressed_DCI = [
        (lossless_compression(args.baseline, s[0], s[1]), s[1]) for s in source_data
    ]

    ## here the stats are collected after all the data have been compressed.
    ## a value of "2" is appended as the null space for the compressed data
    compression_stats = CompressionStatisticCollector(args.baseline)
    for compressed, test_data in compressed_DCI:
        for i in range(len(compressed)):
            compression_stats.add(
                len(compression_stats.stats_df.index) + 1,
                test_data[i],
                compressed[i],
                test_data[i].shape[0] - len(compressed[i]),
            )
    compression_stats.plot_heatmap(saver_directory)