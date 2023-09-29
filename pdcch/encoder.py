from .parameters import ParametersPDCCH
import numpy as np


class PDCCHEncoder(ParametersPDCCH):
    """
    Encode a DCI message according to the preset PDCCH parameters

    Parameters
    ----------
        parameters: object (ParametersPDCCH)
            Inherit all the parameters from the PDCCH parameter setup

    Input
    -----
        [n], numpy.array
            DCI payload message (source, without CRC)

    Output
    -----
        [E], numpy.array
            Encoded DCI message
    """
    def __init__(self, parameters):
        super(PDCCHEncoder, self).__init__()
        self.__dict__.update(parameters.__dict__)

    def __call__(self, DCI_message, *args, **kwargs):
        if len(DCI_message) < 12:
            DCI_message = np.concatenate(
                [DCI_message, np.zeros(12 - len(DCI_message))], axis=0
            )
        self.A = len(DCI_message)
        assert self.A + self.poly == self.K, "A+P should equal K"
        assert np.log2(self.N) == round(np.log2(self.N)), "N should be a power of 2"
        assert (
            sum(self.info_bit_pattern) == self.K
        ), "info_bit_pattern should contain K number of ones."
        assert (
            max(self.rate_matching_pattern) <= self.N
        ), "rate_matching_pattern is not compatible with N"
        assert self.poly >= len(
            self.RNTI
        ), " K should be no less than the length of the scrambing pattern"
        crc_bits = np.mod(np.matmul(DCI_message, self.G_P), 2)

        # Scramble the CRC bits.
        scrambled_crc_bits = np.logical_xor(
            crc_bits,
            np.concatenate([np.zeros((self.poly - len(self.RNTI))), self.RNTI]),
        )

        # Append the scrambled CRC bits to the information bits.
        b = np.concatenate([DCI_message, scrambled_crc_bits])

        # Interleave the information and CRC bits.
        c = b[self.crc_interleaver_pattern]

        # Position the information and CRC bits within the input to the polar
        # encoder kernal.
        u = np.zeros((self.N), dtype=np.int)  ## initialize with all-zeros
        u[self.info_bit_pattern] = c  ## select the most reliable positions

        # Perform the polar encoder kernal operation.
        d = np.mod(np.matmul(u, self.G_N), 2)

        # Extract the encoded bits from the output of the polar encoder kernal.
        e = d[self.rate_matching_pattern]
        return e
