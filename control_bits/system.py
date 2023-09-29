import numpy as np
import random
from .correlated_binary_data import DataGenerator
import tensorflow as tf
import sionna


class UserEquipment:
    """
    Initialize a UserEquipment object that generates packets in the buffer. Given the number of TTIs,
    this object randomly generates a batch of CQI and MCS indices over each RB.

    Parameters
    ----------
        UEs: list of UserEquipment
            List of UserEquipment objects that can generate data packets and report CQIs

        channel_model: object
            A channel model that might be created from Sionna, the channel model should contain
            a function that returns CQI and MCS indices

        poisson_rate: float
            Rate of occurrences in Poisson distribution to generate packets in the buffer

        max_buffer_size: int
            Maximum number of packets that the buffer has

        ebn0: float
            Eb/N0 (dB) of the channel model

        id: int
            Unique Id assigned to a UserEquipment

    """

    def __init__(self, channel_model, poisson_rate, max_buffer_size, ebn0, id):
        super(UserEquipment, self).__init__()
        self._channel_model = channel_model
        self._poisson_rate = poisson_rate
        self._max_buffer_size = max_buffer_size
        self._ebn0 = ebn0
        self._id = id
        self.reset()

    @property
    def id(self):
        return self._id

    @property
    def buffer(self):
        return self._buffer

    @property
    def channel_model(self):
        return self._channel_model

    @property
    def ebn0(self):
        return self._ebn0

    @ebn0.setter
    def ebn0(self, value):
        self._ebn0 = value

    @property
    def poisson_rate(self):
        return self._poisson_rate

    @poisson_rate.setter
    def poisson_rate(self, value):
        self._poisson_rate = value

    def generate_poisson_sample(self):
        """
        Given the time unit length, return the number of packets
        :return: poisson arrival packet at each time index
        """
        return np.random.poisson(self.poisson_rate, 1)[0].astype(int)

    def next(self):
        """generate new packets in the buffer"""
        self._buffer += self.generate_poisson_sample()
        ## clip by the maximum buffer size
        self._buffer = min(self._buffer, self._max_buffer_size)

    def flush_buffer(self, num_packets):
        """clean the buffer according to the given number"""
        self._buffer -= num_packets
        self._buffer = max(self._buffer, 0)

    def renew_RBG_CQI(self):
        ## [batch, TTI, RBs]
        self._CQI, self._MCS = self.channel_model(self.ebn0)[1:]

    def reset(self):
        """assume that the buffer is empty by default"""
        self._buffer = 0


class EvenResourcesNonContiguous:
    """
    Construct a non-contiguous Even Resources scheduler


    Parameters
    ----------
        num_RBs: int
            Defining the number of RBs that the scheduler has

    Input
    -----
        [...], list
            List of UserEquipment candidates
        [], int
            Index of transmission time interval


    Output
    ------
        : {UserEquipment id: numpy.array}, dict
            A dictionary of scheduling decisions (integer indices)

        : {UserEquipment id: int}, dict
            A dictionary of MCS indices

    Note
    ----
    Leftover RBs are allocated to the first "k" shuffled UEs. There are N_l UEs have
    a number of (N_e + 1) RBs

    .. math::
        N_e = floor(N_r / N_u)
        N_l = N_r mod N_u
    """

    def __init__(self, num_RBs):
        self._num_RBs = num_RBs

    def __call__(self, candidates, TTI_idx, *args, **kwargs):
        for c in candidates:
            if c.__class__.__name__ != "UserEquipment":
                raise ValueError("Please check the candidates data type")
        self.scheduling_decision = {}
        self.MCS_decision = {}
        batch_size = candidates[0]._channel_model._batch_size
        num_TD_candidates = sum([c.buffer > 0 for c in candidates])
        if num_TD_candidates == 0:
            return self.scheduling_decision, self.MCS_decision

        ## find the even resources parameters
        num_RBs_per_UE = int(self._num_RBs / num_TD_candidates)
        leftover_RBs = self._num_RBs % num_TD_candidates

        ## shuffle UE candidates indices
        random.shuffle(candidates)

        ## allocate RB, we assume that all the UEs have the same type
        ## there is no weight on the UE type
        ## (batch, RBG index)
        RBs = np.arange(self._num_RBs)
        RBs = np.tile(RBs, (batch_size, 1))

        ## loop over the shuffled UEs and choose even number of RBs for each UE
        for c in candidates:
            num_RBs_to_allocate = num_RBs_per_UE
            if leftover_RBs > 0:
                num_RBs_to_allocate += 1
            CQI_batch_idx = np.tile(
                np.reshape(np.arange(batch_size), (-1, 1)), (1, RBs.shape[1])
            )
            RB_CQI = c._CQI[CQI_batch_idx, TTI_idx, RBs]  # shape: [batch, RB]
            RB_IMCS = c._MCS[CQI_batch_idx, TTI_idx, RBs]
            batch_indices = np.tile(
                np.reshape(np.arange(batch_size), (-1, 1)), (1, num_RBs_to_allocate)
            )
            ## allocate RBs to a UE with the best channel CQIs
            argsort = np.argsort(RB_CQI, axis=1)[:, ::-1][:, :num_RBs_to_allocate]
            self.scheduling_decision[c._id] = RBs[batch_indices, argsort]
            self.MCS_decision[c._id] = np.min(RB_IMCS[batch_indices, argsort], axis=1)

            ## remove RBs that have been allocated
            batch_indices = np.tile(
                np.reshape(np.arange(batch_size), (-1, 1)), (1, num_RBs_to_allocate)
            )
            delete_indices = zip(batch_indices.reshape(-1), argsort.reshape(-1))
            idxs = [i * RBs.shape[1] + j for i, j in delete_indices]
            RBs = np.delete(RBs, idxs).reshape(batch_size, -1)
            if RBs.shape[1] == 0:
                return self.scheduling_decision, self.MCS_decision
            leftover_RBs -= 1
            leftover_RBs = max(leftover_RBs, 0)
        return self.scheduling_decision, self.MCS_decision


def int_bit(integer, num_bits):
    if integer is None:
        return None
    bit_string = bin(integer)[2:].zfill(num_bits)
    return [int(b) for b in bit_string]


def bit_int(bit_array):
    out = 0
    for bit in bit_array:
        out = (out << 1) | bit
    return out


class BTS:
    """
    Initialize a base transceiver station that could generate an epoch of control messages
    depending on the settings of User Equipment, PDCCH preset parameters and system's parameters

    Parameters
    ----------
        UEs: list of UserEquipment
            Defining the list of UserEquipment objects that can generate data packets and report CQIs

        PDCCH_encoder: object of PDCCHEncoder
            Defining the DCI encoding function for a given DCI message

        PDCCH_decoder: object of PDCCHDecoder
            Defining the DCI decoding function for a given sequence of equalized received symbols

        DCI_payload_length: int
            Defining the payload length of a DCI message

        num_RBs: int
            Defining the number of resource blocks that the system support

        num_TTIs: int
            Defining the number of transmission time intervals that 1 epoch contains

        scheduler: object
            Defining the frequency domain scheduling function

        binary_sequence_generator_type: str
            Defining the type of structure to generate the binary variables if DCI payload is larger than
            the sum of scheduling and MCS selection control bits

        activation_ratios: float
            Defining constant float value that is used by the binary sequence generator

        correlation_factor: float
            Defining a constant float value to indicate the correlation factor that is used by the binary sequence generator

        EsN0_control_channel: float
            Defining the SNR level for the control channel

        temporal_memory: int
            Defining the memory buffer that is used to generate features that might be used for machine learning training

    """

    def __init__(
        self,
        UEs,
        PDCCH_encoder,
        PDCCH_decoder,
        DCI_payload_length,
        num_RBs,
        num_TTIs,
        scheduler,
        binary_sequence_generator_type,
        activation_ratios,
        correlation_factor,
        EsN0_control_channel,
        temporal_memory=3,
    ):
        self.UEs = UEs
        self.num_MCS_bits = 5
        self.num_RBs = num_RBs
        self.PDCCH_encoder = PDCCH_encoder
        self.PDCCH_decoder = PDCCH_decoder
        self.scheduler = scheduler
        self.DCI_payload_length = DCI_payload_length
        self.EsN0_control_channel = EsN0_control_channel
        assert DCI_payload_length > (
            self.num_MCS_bits + num_RBs
        ), "DCI payload length needs to be longer"
        self.num_additional_binary = max(
            0, DCI_payload_length - self.num_MCS_bits - num_RBs
        )
        self.binary_sequence_generator = DataGenerator(
            num_TTIs,
            self.num_additional_binary,
            np.array([activation_ratios] * num_TTIs),
            np.array([correlation_factor] * num_TTIs),
            memory_buffer=temporal_memory,
            k_dependent=2,
            type_name=binary_sequence_generator_type,
        )
        self.temporal_memory = temporal_memory
        self.num_TTIs = num_TTIs
        self.num_bits_per_symbol = 2
        self.constellation = sionna.mapping.Constellation(
            "qam", self.num_bits_per_symbol
        )
        self.mapper = sionna.mapping.Mapper(constellation=self.constellation)
        self.reset_DCI_memory()

    def reset(self):
        """reset the buffer and DCI history"""
        [ue.reset() for ue in self.UEs]
        self.reset_DCI_memory()

    def reset_DCI_memory(self):
        """original DCI buffer is set to all -1 (no control message)"""
        self.DCI_memory = {
            ue.id: -1
            * np.ones((self.temporal_memory, self.DCI_payload_length), dtype=np.int32)
            for ue in self.UEs
        }

    def update_UE_control_bits_memory(self, id, control_message):
        """append the latest DCI message to the specific UE Id"""
        ## shape: [TTI, control_message]
        ## roll and replace
        self.DCI_memory[id] = np.roll(self.DCI_memory[id], 1, axis=0)
        self.DCI_memory[id][0, :] = control_message

    def get_FDScheduler_control_bits(self, id, scheduling_decisions):
        if id not in scheduling_decisions.keys():
            return None
        else:
            FDScheduling_control_bits = np.zeros(self.num_RBs, dtype=np.int32)
            FDScheduling_control_bits[scheduling_decisions[id].squeeze()] = 1
            return FDScheduling_control_bits

    def get_MCS_control_bits(self, id, MCS_decisions):
        """use binary sequence to represent the MCS index"""
        MCS_bits = int_bit(MCS_decisions[id].squeeze(), self.num_MCS_bits)
        return MCS_bits

    def generate_new_epoch(self):
        """generate DCI payloads for each UE with shape: (TTI, DCI)"""
        DCI_UE = {
            ue.id: np.empty((self.num_TTIs, self.DCI_payload_length), dtype=np.int32)
            for ue in self.UEs
        }
        encoded_message_UE = {
            ue.id: np.empty((self.num_TTIs, self.PDCCH_encoder.E), dtype=np.int32)
            for ue in self.UEs
        }
        self.reset()

        ## generate new UE channels, shape (num_TTIs, num_RBs)
        [ue.renew_RBG_CQI() for ue in self.UEs]

        ## generate binary bits (batch, num_TTIs, sequence length)
        ## here we assume that the batch dimension refers to the UE dimension
        additional_binary_bits = self.binary_sequence_generator(
            batch_size=len(self.UEs)
        )

        ## randomly generate packets and inherit the random channel and binary bits information
        for t in range(self.num_TTIs):
            ## generate new packets
            [ue.next() for ue in self.UEs]

            ## FD Scheduling
            scheduling_decisions, MCS_selection = self.scheduler(self.UEs, t)

            ## Combine control bits
            for i, u in enumerate(self.UEs):
                scheduling_bits = self.get_FDScheduler_control_bits(
                    u.id, scheduling_decisions
                )
                ## no scheduling command needed
                if scheduling_bits is None:
                    ## no scheduling request
                    DCI_UE[u.id][t] = -1 * np.ones(
                        (self.DCI_payload_length), dtype=np.int32
                    )
                    encoded_message_UE[u.id][t] = -1 * np.ones(
                        (self.PDCCH_encoder.E), dtype=np.int32
                    )
                    self.update_UE_control_bits_memory(u.id, DCI_UE[u.id][t])
                    continue

                MCS_bits = self.get_MCS_control_bits(u.id, MCS_selection)
                DCI_UE[u.id][t] = np.concatenate(
                    [scheduling_bits, MCS_bits, additional_binary_bits[i, t]],
                    axis=-1,
                )
                self.update_UE_control_bits_memory(u.id, DCI_UE[u.id][t])

                encoded_message_UE[u.id][t] = self.PDCCH_encoder(DCI_UE[u.id][t])


                ## decode control message and update the buffer if successfully decoded
                if self.simulate_PDCCH(encoded_message_UE[u.id][t]):
                    u.flush_buffer(sum(scheduling_bits))
        return DCI_UE, encoded_message_UE

    def simulate_PDCCH(self, encoded_DCI_message):
        """assuming the encoded DCI message passes through an AWGN channel"""
        PDCCH_tensor = tf.reshape(
            tf.constant(encoded_DCI_message, dtype=tf.float32),
            (-1, self.num_bits_per_symbol),
        )
        x = self.mapper(PDCCH_tensor)
        ## decoding
        noise_var = tf.convert_to_tensor(
            (1.0 / 10 ** (self.EsN0_control_channel / 10)), dtype=tf.float32
        )
        y = x + (
            np.random.normal(0, 1, x.shape) * np.sqrt(noise_var / 2)
            + 1j * np.random.normal(0, 1, x.shape) * np.sqrt(noise_var / 2)
        )
        y = tf.squeeze(y)
        demapper = sionna.mapping.Demapper("app", constellation=self.constellation)
        llr = demapper([y, noise_var])
        crc_check, a_hat = self.PDCCH_decoder(self.DCI_payload_length, llr)
        return crc_check
