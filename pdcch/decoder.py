from .parameters import ParametersPDCCH
import numpy as np
import tensorflow as tf


class PDCCHDecoder(ParametersPDCCH):
    """
    Decode an encoded DCI message according to the preset PDCCH parameters

    Parameters
    ----------
        parameters: object (ParametersPDCCH)
            Inherit all the parameters from the PDCCH parameter setup

    Input
    -----
        [], int
            DCI payload length
        [E], tf.tensor
            Equalized LLR for each bit of encoded DCI message

    Output
    -----
        [], bool
            True -> CRC check success, False -> CRC check fail
        [A], tf.tensor
            Decoded DCI payload bits

    """
    def __init__(self, parameters):
        super(PDCCHDecoder, self).__init__()
        self.__dict__.update(parameters.__dict__)
        self.llr_threshold = 25
        self.channel_decoder = None

    @property
    def channel_decoder(self):
        return self._channel_decoder

    @channel_decoder.setter
    def channel_decoder(self, _decoder):
        self._channel_decoder = _decoder

    def __call__(self, A, equalized_llr, *args, **kwargs):
        assert A > 0 and A < 140, "Unsupported DCI payload length"
        assert len(equalized_llr) < 8192, "E should be no greater than 8192"
        assert (
            max(self.rate_matching_pattern) <= self.N
        ), "Invalid rate matching pattern"
        assert self.channel_decoder is not None, "Please preset a channel decoder"

        self.A = A  # DCI payload length
        self.E = len(equalized_llr)

        assert np.log2(self.N) == round(np.log2(self.N)), "N should be a power of 2"
        assert (
            sum(self.info_bit_pattern) == self.K
        ), "info_bit_pattern should contain K number of ones."
        assert (
            max(self.rate_matching_pattern) <= self.N
        ), "rate_matching_pattern is not compatible with N"

        if self.mode == "repetition":
            if self.E < self.N:
                raise ValueError("mode is not compatible with E")
        elif self.mode == "puncturing":
            if self.E >= self.N:
                raise ValueError("mode is not compatible with E")
        elif self.mode == "shortening":
            if self.E >= self.N:
                raise ValueError("mode is not compatible with E")
        else:
            raise ValueError("Unsupported mode")

        assert self.poly < len(
            self.crc_polynomial_pattern
        ), "P should be no less than the length of the scrambling pattern"

        # Extend the scrambling pattern to match the length of the CRC
        extended_crc_scrambling_pattern = np.concatenate(
            [np.zeros((self.poly - len(self.RNTI)), dtype=int), self.RNTI], axis=0
        )

        # Rate matching
        if self.mode == "repetition":
            # LLRs for repeated bits are added together.
            mtx_summation = np.zeros((self.E, self.N), dtype=int)

            mtx_summation[
                np.arange(self.E), self.rate_matching_pattern[np.arange(self.E)]
            ] = 1.0
            derate_matching_llr = tf.matmul(
                tf.reshape(equalized_llr, (1, -1)), mtx_summation
            )
        else:
            if self.mode == "puncturing":
                # Zero valued LLRs are used for punctured bits, because the decoder
                # doesn't know if they have values of 0 or 1.
                d_tilde = tf.zeros((self.N), dtype=tf.float32)
            elif self.mode == "shortening":
                # Infinite valued LLRs are used for shortened bits, because the
                # decoder knows that they have values of 0.
                d_tilde = -tf.ones((self.N), dtype=tf.float32) * self.llr_threshold
            else:
                raise ValueError("Unknown rate matching mode")

            derate_matching_llr = tf.tensor_scatter_nd_update(
                d_tilde, self.rate_matching_pattern.reshape(-1, 1), equalized_llr
            )
            derate_matching_llr = tf.expand_dims(derate_matching_llr, axis=0)

        ## acquire decoding results
        decoded_bits = self.channel_decoder(derate_matching_llr)

        ## decoded DCI payload bits
        a_hat = tf.matmul(decoded_bits, self.deinterleaver_matrix).numpy()[0, : self.A]

        ## decramble
        descramble = (
            tf.matmul(decoded_bits, self.deinterleaver_matrix).numpy().astype(int)
        )
        descramble[0, self.A :] = np.bitwise_xor(
            descramble[0, self.A :], extended_crc_scrambling_pattern
        )

        ## re-encode and get the CRC remainder
        crc_remainder = np.mod(np.matmul(descramble, self.G_P2), 2)
        crc_check = tf.reduce_sum(crc_remainder, axis=-1, keepdims=True)
        crc_check = tf.where(crc_check == 0, True, False)

        return tf.squeeze(crc_check), tf.squeeze(a_hat)
