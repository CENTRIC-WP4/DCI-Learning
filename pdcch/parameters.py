import numpy as np


class ParametersPDCCH:
    """
    Initialize the parameters for PDCCH.
    All the codes inherit from the reference Matlab code.

    Parameter
    -----
    E: int
        Encoded DCI message length

    A: int
        DCI payload length

    RNTI: list
        Radio network temporary identifier

    num_bits_per_symbol: int
        Number of bits per symbol



    Ref:
        * https://github.com/vodafone-chair/5g-nr-polar/tree/master
    """

    def __init__(
        self,
        E=128,  # encoded bit length after rate matching
        A=90,  # DCI payload length
        RNTI=[1] * 16,
        num_bits_per_symbol=2,
    ):
        self.E = E
        self.RNTI = RNTI
        self.A = A
        self.num_bits_per_symbol = num_bits_per_symbol


        ## following variables are hard-coded as the information provided below
        self.crc_polynomial_pattern = crc_polynomial_pattern
        self.Pi_IL_max = Pi_IL_max  # interleaver indices
        self.poly = len(self.crc_polynomial_pattern) - 1  # number of CRC bits
        self.Q_Nmax = Q_Nmax  # order of bit-wise capacity in Polar codes

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, value):
        assert value < 8192 and value > 0, "E should be no greater than 8192."
        self._E = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        assert (
            value < 140 and value > 0
        ), "payload length should be greater than 0, but less than 140"
        self._A = value


    @property
    def code_rate(self):
        return self.A / self.E

    @property
    def N(self):
        """determine the number of bits used at the input and output of the polar code kernel"""
        return get_3GPP_N(self.K, self.E, 9)

    @property
    def K(self):
        """payload bits and CRC redundant"""
        if self.A < 12:
            _K = 12 + self.poly
        else:
            _K = self.A + self.poly
        return _K

    @property
    def crc_interleaver_pattern(self):
        """Get the 3GPP CRC interleaver pattern (info & CRC)"""
        return get_3GPP_crc_interleaver_pattern(self.K)

    @property
    def mode(self):
        return get_3GPP_rate_matching_pattern(self.K, self.N, self.E)[1]

    @property
    def rate_matching_pattern(self):
        return get_3GPP_rate_matching_pattern(self.K, self.N, self.E)[0]

    @property
    def Q_N(self):
        """the first element of Q_N gives the index of the least reliable bit and the last element gives the index of the most reliable bit."""
        return get_3GPP_sequence_pattern(self.N)

    @property
    def info_bit_pattern(self):
        return get_3GPP_info_bit_pattern(
            self.K, self.Q_N, self.rate_matching_pattern, self.mode
        )

    @property
    def G_P(self):
        return get_crc_generator_matrix(self.A, self.crc_polynomial_pattern)

    @property
    def G_N(self):
        return get_G_N(self.N)

    @property
    def G_P2(self):
        return get_crc_generator_matrix(self.A + self.poly, self.crc_polynomial_pattern)

    @property
    def deinterleaver_matrix(self):
        interleaver_matrix = np.eye(self.K)
        interleaver_matrix[np.arange(self.K), :] = interleaver_matrix[
            self.crc_interleaver_pattern, :
        ]
        return interleaver_matrix

    @property
    def frozen_bit_positions(self):
        return np.arange(self.N)[self.info_bit_pattern == 0]

    @property
    def info_bit_positions(self):
        return np.arange(self.N)[self.info_bit_pattern]


def get_G_N(N):
    """
    Generate the nth Kronecker power of G_2.

    Input
    -----
    N: int
        Size of N by N matrix, N has to be a power 2

    Output
    -----
    G_N: numpy.array
        N by N matrix for the Polar code generator matrix
    """
    n = int(np.log2(N))
    if n != round(n):
        raise ValueError("N should be a power of 2")

    G_N = 1
    for i in range(n):
        G_N = np.kron(G_N, np.array([[1, 0], [1, 1]]))
    return G_N


def get_crc_generator_matrix(A, crc_polynomial_pattern):
    """
    Generate the generator matrix for a given input message with a length of A

    Input
    -----
        A: int
            Payload length

        crc_polynomial_pattern: list
            Binary sequence of CRC polynomial pattern

    Output
    -----
        G_P: numpy.array
            Generator matrix for the CRC with a shape of [A,P]

    """
    P = len(crc_polynomial_pattern) - 1

    if P < 1:
        raise ValueError("crc_polynomial_pattern is invalid")
    x_crc = np.zeros(P).astype(int)
    G_P = np.zeros([A, P])
    x_crc[0] = 1
    for i in range(A):
        # shift by one position
        x_crc = np.concatenate([x_crc, [0]])
        if x_crc[0] == 1:
            x_crc = np.bitwise_xor(x_crc, crc_polynomial_pattern)
        x_crc = x_crc[1:]
        G_P[A - i - 1, :] = x_crc
    return G_P


def get_3GPP_info_bit_pattern(I, Q_N, rate_matching_pattern, mode):
    """
    Generate a boolean sequence to indicate the positions of information bits

    Input
    -----
        I: int
            Total number of information bits (DCI payload + CRC)

        Q_N: list
            Polar code bit positions with an ascending order of bit-wise capacity

        rate_matching_pattern: list
            Length of E, correspond to the bit positions in the polar encoded sequence

        mode: str
            Mode of rate matching

    Output
    -----
        info_bit_pattern: numpy.array
        Boolean array where True gives the information bits' positions
    """
    N = len(Q_N)
    n = np.log2(N)
    E = len(rate_matching_pattern)

    if n != round(n):
        raise ValueError("N should be a power of 2")

    if I > N:
        raise ValueError(
            "polar_3gpp_matlab:UnsupportedBlockLength", "I should be no greater than N."
        )
    if I > E:
        raise ValueError(
            "polar_3gpp_matlab:UnsupportedBlockLength", "I should be no greater than E."
        )
    if max(rate_matching_pattern) > N:
        raise ValueError("rate_matching_pattern is not compatible with N")
    if mode == "repetition":
        if E < N:
            raise ValueError("mode is not compatible with E")
    elif mode == "puncturing":
        if E >= N:
            raise ValueError("mode is not compatible with E")
    elif mode == "shortening":
        if E >= N:
            raise ValueError("mode is not compatible with E")
    else:
        raise ValueError("Unsupported mode")

    ## pre-frozen bits, not in the rate-matching pattern
    Q_Ftmp_N = np.setdiff1d(np.arange(N), rate_matching_pattern)


    if mode == "puncturing":
        if E >= 3 * N / 4:
            Q_Ftmp_N = np.concatenate(
                [Q_Ftmp_N, np.arange(int(np.ceil(3 * N / 4 - E / 2)))]
            )
        else:
            Q_Ftmp_N = np.concatenate(
                [Q_Ftmp_N, np.arange(int(np.ceil(9 * N / 16 - E / 4)))]
            )

    ## order matters, Itmp -> actual information position
    Q_Itmp_N = np.array([el for el in Q_N if el not in Q_Ftmp_N], dtype=int)

    if len(Q_Itmp_N) < I:
        raise ValueError("Too many pre-frozen bits")

    ## choose the positions with the highest reliability
    Q_I_N = Q_Itmp_N[-I:]
    info_bit_pattern = np.array([False] * N, dtype=bool)

    info_bit_pattern[Q_I_N] = True
    return info_bit_pattern


def get_3GPP_sequence_pattern(N):
    """
    Generate an array with integers to indicate the order of bit-wise capacities of Polar code
    bit positions. The higher of the index of the returned array, the greater the bit-wise capacity

    Input
    -----
        N: int
            Polar code generator matrix size

    Output
    -----
        Q_N: array of integer indices, where the index of array follows an ascending order of bit-wise capacity

    """
    if np.log2(N) != round(np.log2(N)):
        raise ValueError("N should be a power of 2")

    if N > len(Q_Nmax):
        raise ValueError("Value of N is unsupported")

    Q_N = Q_Nmax[Q_Nmax < N]
    return Q_N


def get_3GPP_rate_matching_pattern(K, N, E):
    """
    Generate a rate-matching pattern that contains the integer indices, which correspond to the bit
    position in the Polar code generator matrix.

    Input
    -----
        K: int
            Number of information bits plus CRC

        N: int
            Polar code generator matrix size

        E: int
            Encoded DCI message length

    Output
    -----
        rate_matching_pattern: numpy.array
            Integer indices that range from 0 to N-1, which correspond to the bit indices from a Polar code

        mode: str
            Rate matching mode ("Repetition", "Shorting", "Puncturing")

    """
    n = np.log2(N)
    assert not (n != round(n))
    assert not (n < 5)

    d = np.arange(N)

    J = np.zeros((N), dtype=int)
    y = np.zeros((N), dtype=int)
    for n in range(0, N):
        i = np.floor(32 * n / N).astype(int)
        J[n] = int(P[i] * (N / 32)) + int(np.mod(n, int(N / 32)))
        y[n] = d[J[n]]
    rate_matching_pattern = np.zeros(E, dtype=int)
    if E >= N:
        for k in range(0, E):
            rate_matching_pattern[k] = y[np.mod(k, N)]
        mode = "repetition"
    else:
        if K / E <= 7 / 16:
            for k in range(0, E):
                rate_matching_pattern[k] = y[k + N - E]
            mode = "puncturing"
        else:
            for k in range(0, E):
                rate_matching_pattern[k] = y[k]
            mode = "shortening"

    return rate_matching_pattern, mode


def get_3GPP_N(K, E, n_max):
    """
    Obtain the required Polar code size to satisfy the given number of (DCI payload + CRC bits) and
    encoded DCI message length. Polar code generator matrix should not exceed a size of 2 ** n_max

    Input
    -----
        K: int
            number of information bits plus CRC bits

        E: int
            encoded DCI message length

        n_max: int
            maximum order of Polar code

    Output
    -----
        N: int
            Size of Polar code generator matrix
    """

    if E <= (9 / 8) * 2 ** (np.ceil(np.log2(E)) - 1) and K / E < 9 / 16:
        n_1 = int(np.ceil(np.log2(E))) - 1
    else:
        n_1 = np.ceil(np.log2(E))

    R_min = 1 / 8
    n_min = 5
    n_2 = np.ceil(np.log2(K / R_min))
    n = max(n_min, min([n_1, n_2, n_max]))

    N = int(2**n)  ## minimum length 2**5 = 32
    return N


def get_3GPP_crc_interleaver_pattern(K):
    """
    Generate the interleaver pattern for the combined sequence of DCI payload and CRC

    Input
    -----
        K: int
            Number of bits of combined sequence of DCI payload and CRC

    Output
    -----
        Pi: numpy.array
            Array of integer values that interleave a binary sequence with a number of K elements

    """

    if K > len(Pi_IL_max):
        raise ValueError(
            "polar_3gpp_matlab:UnsupportedBlockLength",
            "K should be no greater than 224.",
        )

    Pi = np.zeros((K), dtype=int)
    k = 0
    for m in range(len(Pi_IL_max)):
        if Pi_IL_max[m] >= len(Pi_IL_max) - K:
            Pi[k] = Pi_IL_max[m] - (len(Pi_IL_max) - K)
            k = k + 1
    return Pi


# The CRC polynomial used in 3GPP PBCH and PDCCH channel is
# D^24 + D^23 + D^21 + D^20 + D^17 + D^15 + D^13 + D^12 + D^8 + D^4 + D^2 + D + 1
crc_polynomial_pattern = np.array(
    [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
)

Pi_IL_max = [
    0,
    2,
    3,
    5,
    6,
    8,
    11,
    12,
    13,
    16,
    19,
    20,
    22,
    24,
    28,
    32,
    33,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    46,
    47,
    49,
    50,
    54,
    55,
    57,
    59,
    60,
    62,
    64,
    67,
    69,
    74,
    79,
    80,
    84,
    85,
    86,
    88,
    91,
    94,
    102,
    105,
    109,
    110,
    111,
    113,
    114,
    116,
    118,
    119,
    121,
    122,
    125,
    126,
    127,
    129,
    130,
    131,
    132,
    136,
    137,
    141,
    142,
    143,
    147,
    148,
    149,
    151,
    153,
    155,
    158,
    161,
    164,
    166,
    168,
    170,
    171,
    173,
    175,
    178,
    179,
    180,
    182,
    183,
    186,
    187,
    189,
    192,
    194,
    198,
    199,
    200,
    1,
    4,
    7,
    9,
    14,
    17,
    21,
    23,
    25,
    29,
    34,
    36,
    43,
    45,
    48,
    51,
    56,
    58,
    61,
    63,
    65,
    68,
    70,
    75,
    81,
    87,
    89,
    92,
    95,
    103,
    106,
    112,
    115,
    117,
    120,
    123,
    128,
    133,
    138,
    144,
    150,
    152,
    154,
    156,
    159,
    162,
    165,
    167,
    169,
    172,
    174,
    176,
    181,
    184,
    188,
    190,
    193,
    195,
    201,
    10,
    15,
    18,
    26,
    30,
    52,
    66,
    71,
    76,
    82,
    90,
    93,
    96,
    104,
    107,
    124,
    134,
    139,
    145,
    157,
    160,
    163,
    177,
    185,
    191,
    196,
    202,
    27,
    31,
    53,
    72,
    77,
    83,
    97,
    108,
    135,
    140,
    146,
    197,
    203,
    73,
    78,
    98,
    204,
    99,
    205,
    100,
    206,
    101,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
]

P = np.array(
    [
        0,
        1,
        2,
        4,
        3,
        5,
        6,
        7,
        8,
        16,
        9,
        17,
        10,
        18,
        11,
        19,
        12,
        20,
        13,
        21,
        14,
        22,
        15,
        23,
        24,
        25,
        26,
        28,
        27,
        29,
        30,
        31,
    ]
)

Q_Nmax = np.array(
    [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        3,
        5,
        64,
        9,
        6,
        17,
        10,
        18,
        128,
        12,
        33,
        65,
        20,
        256,
        34,
        24,
        36,
        7,
        129,
        66,
        512,
        11,
        40,
        68,
        130,
        19,
        13,
        48,
        14,
        72,
        257,
        21,
        132,
        35,
        258,
        26,
        513,
        80,
        37,
        25,
        22,
        136,
        260,
        264,
        38,
        514,
        96,
        67,
        41,
        144,
        28,
        69,
        42,
        516,
        49,
        74,
        272,
        160,
        520,
        288,
        528,
        192,
        544,
        70,
        44,
        131,
        81,
        50,
        73,
        15,
        320,
        133,
        52,
        23,
        134,
        384,
        76,
        137,
        82,
        56,
        27,
        97,
        39,
        259,
        84,
        138,
        145,
        261,
        29,
        43,
        98,
        515,
        88,
        140,
        30,
        146,
        71,
        262,
        265,
        161,
        576,
        45,
        100,
        640,
        51,
        148,
        46,
        75,
        266,
        273,
        517,
        104,
        162,
        53,
        193,
        152,
        77,
        164,
        768,
        268,
        274,
        518,
        54,
        83,
        57,
        521,
        112,
        135,
        78,
        289,
        194,
        85,
        276,
        522,
        58,
        168,
        139,
        99,
        86,
        60,
        280,
        89,
        290,
        529,
        524,
        196,
        141,
        101,
        147,
        176,
        142,
        530,
        321,
        31,
        200,
        90,
        545,
        292,
        322,
        532,
        263,
        149,
        102,
        105,
        304,
        296,
        163,
        92,
        47,
        267,
        385,
        546,
        324,
        208,
        386,
        150,
        153,
        165,
        106,
        55,
        328,
        536,
        577,
        548,
        113,
        154,
        79,
        269,
        108,
        578,
        224,
        166,
        519,
        552,
        195,
        270,
        641,
        523,
        275,
        580,
        291,
        59,
        169,
        560,
        114,
        277,
        156,
        87,
        197,
        116,
        170,
        61,
        531,
        525,
        642,
        281,
        278,
        526,
        177,
        293,
        388,
        91,
        584,
        769,
        198,
        172,
        120,
        201,
        336,
        62,
        282,
        143,
        103,
        178,
        294,
        93,
        644,
        202,
        592,
        323,
        392,
        297,
        770,
        107,
        180,
        151,
        209,
        284,
        648,
        94,
        204,
        298,
        400,
        608,
        352,
        325,
        533,
        155,
        210,
        305,
        547,
        300,
        109,
        184,
        534,
        537,
        115,
        167,
        225,
        326,
        306,
        772,
        157,
        656,
        329,
        110,
        117,
        212,
        171,
        776,
        330,
        226,
        549,
        538,
        387,
        308,
        216,
        416,
        271,
        279,
        158,
        337,
        550,
        672,
        118,
        332,
        579,
        540,
        389,
        173,
        121,
        553,
        199,
        784,
        179,
        228,
        338,
        312,
        704,
        390,
        174,
        554,
        581,
        393,
        283,
        122,
        448,
        353,
        561,
        203,
        63,
        340,
        394,
        527,
        582,
        556,
        181,
        295,
        285,
        232,
        124,
        205,
        182,
        643,
        562,
        286,
        585,
        299,
        354,
        211,
        401,
        185,
        396,
        344,
        586,
        645,
        593,
        535,
        240,
        206,
        95,
        327,
        564,
        800,
        402,
        356,
        307,
        301,
        417,
        213,
        568,
        832,
        588,
        186,
        646,
        404,
        227,
        896,
        594,
        418,
        302,
        649,
        771,
        360,
        539,
        111,
        331,
        214,
        309,
        188,
        449,
        217,
        408,
        609,
        596,
        551,
        650,
        229,
        159,
        420,
        310,
        541,
        773,
        610,
        657,
        333,
        119,
        600,
        339,
        218,
        368,
        652,
        230,
        391,
        313,
        450,
        542,
        334,
        233,
        555,
        774,
        175,
        123,
        658,
        612,
        341,
        777,
        220,
        314,
        424,
        395,
        673,
        583,
        355,
        287,
        183,
        234,
        125,
        557,
        660,
        616,
        342,
        316,
        241,
        778,
        563,
        345,
        452,
        397,
        403,
        207,
        674,
        558,
        785,
        432,
        357,
        187,
        236,
        664,
        624,
        587,
        780,
        705,
        126,
        242,
        565,
        398,
        346,
        456,
        358,
        405,
        303,
        569,
        244,
        595,
        189,
        566,
        676,
        361,
        706,
        589,
        215,
        786,
        647,
        348,
        419,
        406,
        464,
        680,
        801,
        362,
        590,
        409,
        570,
        788,
        597,
        572,
        219,
        311,
        708,
        598,
        601,
        651,
        421,
        792,
        802,
        611,
        602,
        410,
        231,
        688,
        653,
        248,
        369,
        190,
        364,
        654,
        659,
        335,
        480,
        315,
        221,
        370,
        613,
        422,
        425,
        451,
        614,
        543,
        235,
        412,
        343,
        372,
        775,
        317,
        222,
        426,
        453,
        237,
        559,
        833,
        804,
        712,
        834,
        661,
        808,
        779,
        617,
        604,
        433,
        720,
        816,
        836,
        347,
        897,
        243,
        662,
        454,
        318,
        675,
        618,
        898,
        781,
        376,
        428,
        665,
        736,
        567,
        840,
        625,
        238,
        359,
        457,
        399,
        787,
        591,
        678,
        434,
        677,
        349,
        245,
        458,
        666,
        620,
        363,
        127,
        191,
        782,
        407,
        436,
        626,
        571,
        465,
        681,
        246,
        707,
        350,
        599,
        668,
        790,
        460,
        249,
        682,
        573,
        411,
        803,
        789,
        709,
        365,
        440,
        628,
        689,
        374,
        423,
        466,
        793,
        250,
        371,
        481,
        574,
        413,
        603,
        366,
        468,
        655,
        900,
        805,
        615,
        684,
        710,
        429,
        794,
        252,
        373,
        605,
        848,
        690,
        713,
        632,
        482,
        806,
        427,
        904,
        414,
        223,
        663,
        692,
        835,
        619,
        472,
        455,
        796,
        809,
        714,
        721,
        837,
        716,
        864,
        810,
        606,
        912,
        722,
        696,
        377,
        435,
        817,
        319,
        621,
        812,
        484,
        430,
        838,
        667,
        488,
        239,
        378,
        459,
        622,
        627,
        437,
        380,
        818,
        461,
        496,
        669,
        679,
        724,
        841,
        629,
        351,
        467,
        438,
        737,
        251,
        462,
        442,
        441,
        469,
        247,
        683,
        842,
        738,
        899,
        670,
        783,
        849,
        820,
        728,
        928,
        791,
        367,
        901,
        630,
        685,
        844,
        633,
        711,
        253,
        691,
        824,
        902,
        686,
        740,
        850,
        375,
        444,
        470,
        483,
        415,
        485,
        905,
        795,
        473,
        634,
        744,
        852,
        960,
        865,
        693,
        797,
        906,
        715,
        807,
        474,
        636,
        694,
        254,
        717,
        575,
        913,
        798,
        811,
        379,
        697,
        431,
        607,
        489,
        866,
        723,
        486,
        908,
        718,
        813,
        476,
        856,
        839,
        725,
        698,
        914,
        752,
        868,
        819,
        814,
        439,
        929,
        490,
        623,
        671,
        739,
        916,
        463,
        843,
        381,
        497,
        930,
        821,
        726,
        961,
        872,
        492,
        631,
        729,
        700,
        443,
        741,
        845,
        920,
        382,
        822,
        851,
        730,
        498,
        880,
        742,
        445,
        471,
        635,
        932,
        687,
        903,
        825,
        500,
        846,
        745,
        826,
        732,
        446,
        962,
        936,
        475,
        853,
        867,
        637,
        907,
        487,
        695,
        746,
        828,
        753,
        854,
        857,
        504,
        799,
        255,
        964,
        909,
        719,
        477,
        915,
        638,
        748,
        944,
        869,
        491,
        699,
        754,
        858,
        478,
        968,
        383,
        910,
        815,
        976,
        870,
        917,
        727,
        493,
        873,
        701,
        931,
        756,
        860,
        499,
        731,
        823,
        922,
        874,
        918,
        502,
        933,
        743,
        760,
        881,
        494,
        702,
        921,
        501,
        876,
        847,
        992,
        447,
        733,
        827,
        934,
        882,
        937,
        963,
        747,
        505,
        855,
        924,
        734,
        829,
        965,
        938,
        884,
        506,
        749,
        945,
        966,
        755,
        859,
        940,
        830,
        911,
        871,
        639,
        888,
        479,
        946,
        750,
        969,
        508,
        861,
        757,
        970,
        919,
        875,
        862,
        758,
        948,
        977,
        923,
        972,
        761,
        877,
        952,
        495,
        703,
        935,
        978,
        883,
        762,
        503,
        925,
        878,
        735,
        993,
        885,
        939,
        994,
        980,
        926,
        764,
        941,
        967,
        886,
        831,
        947,
        507,
        889,
        984,
        751,
        942,
        996,
        971,
        890,
        509,
        949,
        973,
        1000,
        892,
        950,
        863,
        759,
        1008,
        510,
        979,
        953,
        763,
        974,
        954,
        879,
        981,
        982,
        927,
        995,
        765,
        956,
        887,
        985,
        997,
        986,
        943,
        891,
        998,
        766,
        511,
        988,
        1001,
        951,
        1002,
        893,
        975,
        894,
        1009,
        955,
        1004,
        1010,
        957,
        983,
        958,
        987,
        1012,
        999,
        1016,
        767,
        989,
        1003,
        990,
        1005,
        959,
        1011,
        1013,
        895,
        1006,
        1014,
        1017,
        1018,
        991,
        1020,
        1007,
        1015,
        1019,
        1021,
        1022,
        1023,
    ]
)
