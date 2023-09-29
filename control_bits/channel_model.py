import numpy as np
import sionna
import tensorflow as tf
from sionna.mimo import StreamManagement
from sionna.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
)
from sionna.ofdm import (
    OFDMModulator,
    OFDMDemodulator,
    ZFPrecoder,
    RemoveNulledSubcarriers,
)
from sionna.channel.tr38901 import AntennaArray, CDL
from sionna.channel import (
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)
from sionna.channel import ApplyTimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no

sionna.config.xla_compat = True


class CQIAndMCSSelection:
    """
    Select CQI and MCS indices according to measured SINR

    Input
    -----
        [...], numpy.array
            Measured SINR

    Output
    -----
        [...], numpy.array
            CQI indices that have the same shape as input SINR

        [...], numpy.array
            MCS indices that have the same shape as input SINR

    Ref:
    * https://uk.mathworks.com/help/5g/ug/5g-nr-downlink-csi-reporting.html
    """

    def __init__(self):
        p = np.array([2.11, -9.24])
        self.SINRs90pc = np.polyval(
            p, np.concatenate([np.array([2]), np.arange(2, 16)], axis=0)
        )

    def CQI_MCS_Selection(self, SINR_measured):
        """return same shape as input with CQI index and IMCS indices"""
        num_elements = SINR_measured.size
        original_shape = SINR_measured.shape
        SINR_measured = SINR_measured.reshape(-1)
        IMCS = np.empty(num_elements, dtype=np.int32)
        CQI_index = np.empty(num_elements, dtype=np.int32)

        def find_cqi_imcs(sinr):
            cqi = np.argwhere(self.SINRs90pc < sinr)
            IMCSTable = [-1, 0, 0, 2, 4, 6, 8, 11, 13, 16, 18, 21, 23, 25, 27, 27]
            if len(cqi) == 0:
                CQI_index = 1
            else:
                CQI_index = cqi[-1][0] + 1
            IMCS = IMCSTable[CQI_index]
            return CQI_index, IMCS

        for i in range(num_elements):
            CQI_index[i], IMCS[i] = find_cqi_imcs(SINR_measured[i])
        return CQI_index.reshape(original_shape), IMCS.reshape(original_shape)


class CDLModel(tf.keras.Model):
    """
    A CDL model with LDPC encoder, CDL channel model and MIMO configuration.
    All the details can be found in Sionna's tutorial

    Parameters
    -----
    cdl_model_type: str
        CDL model to use. Must be one of "A", "B", "C", "D" or "E".

    delay_spread: float
        RMS delay spread [s].

    perfect_csi: bool
        Defining whether to assume perfect CSI

    speed: float
        Minimum speed [m/s]. Defaults to 0.

    cyclic_prefix_length: int
        Length of the cyclic prefix.

    num_TTIs: int
        Number of transmission time intervals

    num_RBs: int
        Number of resource blocks

    batch_size: int
        Batch size for generating the channel state information

    direction : str
        Link direction. Must be either "uplink" or "downlink".

    subcarrier_spacing : float
        The subcarrier spacing in Hz.

    carrier_frequency : float
        Carrier frequency [Hz]

    Input
    -----
    ebn0_db: float
        noise level for AWGN in the channel model

    Ouptut
    -----
    SINR: numpy.array
        Signal to Interference & Noise Ratio values with shape: (batch_size, num_TTIs, num_RBs)

    CQI: numpy.array
        Channel quality indicator (batch_size, num_TTIs, num_RBs)

    MCS: numpy.array
        Modulation and coding scheme indices (batch, num_TTIs, num_RBs)

    Ref:
    * https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html
    """

    def __init__(
        self,
        cdl_model_type,
        delay_spread,
        perfect_csi,
        speed,
        cyclic_prefix_length,
        num_TTIs,
        num_RBs,
        batch_size=1, # set to 1 since the buffer is updated after each TTI
        direction="downlink",
        subcarrier_spacing=15e3,
        carrier_frequency=2.6e9,
    ):
        super().__init__()

        self.cqi_mcs = CQIAndMCSSelection()

        self._batch_size = batch_size

        # Provided parameters
        self._direction = direction
        self._cdl_model = cdl_model_type
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length

        # System parameters
        self._carrier_frequency = carrier_frequency
        self._subcarrier_spacing = subcarrier_spacing
        self._num_RBs = num_RBs
        self._fft_per_RB = 12
        self._fft_size = num_RBs * self._fft_per_RB
        self._num_ofdm_symbols = 14
        self._num_TTIs = num_TTIs
        self._num_ut = 1
        self._num_bs = 1
        self._num_ut_ant = 2
        self._num_bs_ant = 2
        self._num_bits_per_symbol = 2
        self._coderate = 0.5
        self._pilot_pattern = "kronecker"
        self._rx_tx_association = np.array([[1]])
        self._num_streams_per_tx = self._num_ut_ant
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        self._ut_array = AntennaArray(
            num_rows=1,
            num_cols=int(self._num_ut_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self._carrier_frequency,
        )
        self._bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(self._num_bs_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self._carrier_frequency,
        )

        self._cdl = CDL(
            model=self._cdl_model,
            delay_spread=self._delay_spread,
            carrier_frequency=self._carrier_frequency,
            ut_array=self._ut_array,
            bs_array=self._bs_array,
            direction=self._direction,
            min_speed=self._speed,
        )

        ## assume that pilot symbols are located at 2nd and 11th symbol indices
        self._pilot_ofdm_symbol_indices = [
            i * self._num_ofdm_symbols + 2 for i in range(num_TTIs)
        ]
        self._pilot_ofdm_symbol_indices += [
            i * self._num_ofdm_symbols + 11 for i in range(num_TTIs)
        ]
        self._rg = ResourceGrid(
            num_ofdm_symbols=self._num_ofdm_symbols * num_TTIs,
            fft_size=self._fft_size,
            subcarrier_spacing=self._subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self._num_streams_per_tx,
            cyclic_prefix_length=self._cyclic_prefix_length,
            num_guard_carriers=[5, 6],
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices,
        )
        self._frequencies = subcarrier_frequencies(
            self._rg.fft_size, self._rg.subcarrier_spacing
        )
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
        self._l_tot = self._l_max - self._l_min + 1
        self._channel_time = ApplyTimeChannel(
            self._rg.num_time_samples, l_tot=self._l_tot, add_awgn=True
        )
        self._modulator = OFDMModulator(self._cyclic_prefix_length)
        self._demodulator = OFDMDemodulator(
            self._fft_size, self._l_min, self._cyclic_prefix_length
        )
        self._binary_source = BinarySource()
        self._n = int(
            self._rg.num_data_symbols / self._num_TTIs * self._num_bits_per_symbol
        )  # Number of coded bits
        self._k = int(self._n * self._coderate)  # Number of information bits
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._zf_precoder = ZFPrecoder(
            self._rg, self._sm, return_effective_channel=True
        )
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

        ## initialize the RB indices and OFDM symbol indices for SINR collection
        self._RB_subcarrier_ind = {}
        for i in self._rg.effective_subcarrier_ind:
            if int(i / self._fft_per_RB) not in self._RB_subcarrier_ind.keys():
                self._RB_subcarrier_ind[int(i / self._fft_per_RB)] = []
            self._RB_subcarrier_ind[int(i / self._fft_per_RB)].append(i)

        self._TTI_symbol_index = {
            i: np.arange(i * self._num_ofdm_symbols, (i + 1) * self._num_ofdm_symbols)
            for i in range(self._num_TTIs)
        }

    @property
    def num_TTIs(self):
        return self._num_TTIs

    @num_TTIs.setter
    def num_TTIs(self, new_value):
        self._num_TTIs = new_value

    def __call__(self, ebno_db):
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        x_concat = []
        for _ in range(self._num_TTIs):
            b = self._binary_source(
                [self._batch_size, 1, self._rg.num_streams_per_tx, self._encoder.k]
            )
            c = self._encoder(b)
            x = self._mapper(c)
            x_concat.append(x)
        x_concat = tf.concat(x_concat, axis=-1)
        x_rg = self._rg_mapper(x_concat)

        while 1:
            try:
                ## in case that the precoder matrix is not invertable
                a, tau = self._cdl(
                    self._batch_size,
                    self._rg.num_time_samples + self._l_tot - 1,
                    self._rg.bandwidth,
                )
                h_time = cir_to_time_channel(
                    self._rg.bandwidth,
                    a,
                    tau,
                    l_min=self._l_min,
                    l_max=self._l_max,
                    normalize=True,
                )

                a_freq = a[
                    ...,
                    self._rg.cyclic_prefix_length : -1 : (
                        self._rg.fft_size + self._rg.cyclic_prefix_length
                    ),
                ]
                a_freq = a_freq[..., : self._rg.num_ofdm_symbols]
                h_freq = cir_to_ofdm_channel(
                    self._frequencies, a_freq, tau, normalize=True
                )
                x_rg, g = self._zf_precoder([x_rg, h_freq])
                break
            except:
                pass
        x_time = self._modulator(x_rg)
        y_time = self._channel_time([x_time, h_time, no])
        y = self._demodulator(y_time)

        if self._perfect_csi:
            h_hat, err_var = g, 0.0
        else:
            h_hat, err_var = self._ls_est([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        no_eff_map = self._rg_mapper(tf.cast(no_eff, tf.complex64))

        ## no_eff_map shape: (batch, tx, rx, tti, rb)
        report_shape = (self._batch_size, self._num_TTIs, self._num_RBs)
        SINR = np.empty(report_shape, dtype=np.float32)
        CQI, MCS = np.empty(report_shape, dtype=np.float32), np.empty(
            report_shape, dtype=np.int32
        )
        no_eff_numpy = np.abs(no_eff_map.numpy())
        for i in range(self._num_TTIs):
            for j in range(self._num_RBs):
                TTI_symbol_ind = self._TTI_symbol_index[i]
                RB_subcarrier_ind = self._RB_subcarrier_ind[j]
                SINR[..., i, j] = 10.0 * np.log(
                    1.0
                    / no_eff_numpy[
                        ...,
                        TTI_symbol_ind[0] : TTI_symbol_ind[-1] + 1,
                        RB_subcarrier_ind[0] : RB_subcarrier_ind[-1] + 1,
                    ].mean(axis=(1, 2, 3, 4)),
                )
                CQI[..., i, j], MCS[..., i, j] = self.cqi_mcs.CQI_MCS_Selection(
                    SINR[..., i, j]
                )
        return SINR, CQI, MCS
