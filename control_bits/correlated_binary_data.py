import numpy as np


class BinaryCorrelatedData:
    """BinaryCorrelatedData(t, n, dtype=np.float32)

    Generate binary data that are temporally correlated.

    This class randomly generates binary bits that are correlated over the given ``t`` time indices
    and the `n` number of bits

    Parameters
    ----------
        t: int
            Defining the number of time indices that the bits are correlated at.

        n: int
            Defining the number of binary variables.

        dtype: np.DType
            Defaults to `np.float32`. Defines the output datatype.

    Raises
    ------
        ValueError
            If ``k`` is not a positive integer.

        ValueError
            If ``n`` is not a positive integer.

        ValueError
            If ``dtype`` is not supported.

    Note
    ----
        Correlation is assumed to be existed in the time-domain.
        In the spatial domain, binary bits are independent.
    """

    def __init__(self, t, n, dtype=np.float32):
        if dtype not in (
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
        ):
            raise ValueError("Unsupported dtype.")
        assert t > 0, "negative dimensions are not allowed"
        assert n > 0, "negative dimensions are not allowed"
        self._t = t
        self._n = n
        self.dtype = dtype

    #########################################
    # Public methods and properties
    #########################################

    @property
    def t(self):
        """Number of time indices."""
        return self._t

    @t.setter
    def t(self, new_value):
        self._t = new_value

    @property
    def n(self):
        """Number of binary bits"""
        return self._n

    @n.setter
    def n(self, new_value):
        self._n = new_value


class DecayingProductCorrelation(BinaryCorrelatedData):
    """DecayingProductCorrelation(t, n, dtype=np.float32)

    Generate binary data that are temporally correlated.

    This class randomly generates binary bits that are correlated over the given ``t`` time indices
    and the `n` number of bits.

    Parameters
    ----------
        t: int
            Defining the number of time indices that the bits are correlated at.

        n: int
            Defining the number of binary variables.

        dtype: np.DType
            Defaults to `np.float32`. Defines the output datatype.

    Input
    -----
        [..., t], np.float32
            Activation ratios for Bernoulli random variables
        [..., t], np.float32
            Off-diagonal correlation array for the `t` variables


    Output
    ------
        : [t, n], np.float32
            2D array containing the correlated binary bits with a shape of [t, n].

    Raises
    ------
        ValueError
            If inputs lengths are greater than `k`.


    Note
    ----
    Correlation is assumed to be existed in the time-domain.
    In the spatial domain, binary bits are independent.

    .. math::
            ┌ 1                          r1                r1r2  ... \prod_{l=1}^{k-1}r_l ┐
        R = │ r1                          1                r2    ... \prod_{l=2}^{k-1}r_l │
            | ...                        ...              ...  ...             :          |
            └ \prod_{l=1}^{k-1}r_l  \prod_{l=2}^{k-1}r_l  ...  ...             1          ┘

    Ref:
    * Wei Jiang, Shuang Song, Lin Hou & Hongyu Zhao (2021)
    A Set of Efficient Methods to Generate High-Dimensional Binary Data With Specified Correlation Structures,
    The American Statistician, 75:3, 310-322,
    """

    def __init__(self, t, n, dtype=np.float32):
        super().__init__(t, n, dtype=dtype)

    def __call__(self, activation_ratios, off_diagonal_correlation, batch_size=3):
        assert isinstance(
            activation_ratios, np.ndarray
        ), "inputs should be numpy.ndarray"
        assert (
            activation_ratios.shape[-1] == self.t
        ), "activation ratio shape does not match preset"
        assert isinstance(
            off_diagonal_correlation, np.ndarray
        ), "off-dignonal correlation should be numpy.ndarray"
        assert (
            off_diagonal_correlation.shape[-1] == self.t
        ), "off-dignonal correlation shape does not match preset"

        correlated_output = np.empty((batch_size, self.t, self.n), dtype=np.float32)
        correlated_output[:, 0] = np.random.binomial(
            1, activation_ratios[0], (batch_size, self.n)
        )
        for i in range(1, self.t):
            rho_ = off_diagonal_correlation[i - 1]
            p_i = activation_ratios[i]
            p_i_1 = activation_ratios[i - 1]
            alpha = rho_ * np.sqrt(p_i * (1 - p_i) / (p_i_1 * (1.0 - p_i_1)))
            beta = (p_i - alpha * p_i_1) / (1.0 - alpha)
            U = np.random.binomial(1, alpha, (batch_size, self.n))
            Y = np.random.binomial(1, beta, (batch_size, self.n))
            correlated_output[:, i] = (1.0 - U) * Y + U * correlated_output[:, i - 1]
        return correlated_output.astype(self.dtype)


class OneDepedentCorrelation(BinaryCorrelatedData):
    """OneDepedentCorrelation(t, n, dtype=np.float32)

    Generate binary data that are temporally correlated.

    This class randomly generates binary bits that are correlated over the given ``t`` time indices
    and the `n` number of bits.

    Parameters
    ----------
        t: int
            Defining the number of time indices that the bits are correlated at.

        n: int
            Defining the number of binary variables.

        dtype: np.DType
            Defaults to `np.float32`. Defines the output datatype.

    Input
    -----
        [..., t], np.float32
            Activation ratios for Bernoulli random variables
        [..., t], np.float32
            Correlation coefficient vector for the `t` variables


    Output
    ------
        : [t, n], np.float32
            2D array containing the correlated binary bits with a shape of [t, n].

    Raises
    ------
        ValueError
            If inputs lengths are greater than `k`.


    Note
    ----
    Correlation is assumed to be existed in the time-domain.
    In the spatial domain, binary bits are independent.

    .. math::
            ┌ 1     r1      0  ... 0 ┐
        R = │ r1    1       r2 ... 0 │
            │ ...  ...     ... ... : │
            └ 0     0 ...  ... ... 1 ┘

    Ref:
    * Wei Jiang, Shuang Song, Lin Hou & Hongyu Zhao (2021)
    A Set of Efficient Methods to Generate High-Dimensional Binary Data With Specified Correlation Structures,
    The American Statistician, 75:3, 310-322,
    """

    def __init__(self, t, n, dtype=np.float32):
        super().__init__(t, n, dtype=dtype)

    def prentice_constrains(self, activation_ratios, correlation_coefficients):
        """Assume that correlation_coefficients is 1 dependent"""
        for i, c in enumerate(correlation_coefficients):
            p_i = activation_ratios[i]
            p_j = activation_ratios[i + 1]
            left_factor = np.sqrt(p_i * (1.0 - p_j) / (p_j * (1.0 - p_i)))
            right_factor = np.sqrt(p_j * (1.0 - p_i) / (p_i * (1.0 - p_j)))
            if max(-left_factor, -right_factor) > c:
                return False
            elif min(left_factor, right_factor) < c:
                return False
        return True

    def __call__(self, activation_ratios, correlation_coefficients, batch_size=3):
        assert isinstance(
            activation_ratios, np.ndarray
        ), "inputs should be numpy.ndarray"
        assert (
            activation_ratios.shape[-1] == self.t
        ), "activation ratio shape does not match preset"
        assert isinstance(
            correlation_coefficients, np.ndarray
        ), "correlation coefficients should be numpy.ndarray"
        assert (
            correlation_coefficients.shape[-1] == self.t - 1
        ), "correlation coefficients' shape should be 1 less than number of time indices (1-dependent)"
        assert self.prentice_constrains(
            activation_ratios, correlation_coefficients
        ), "Prentice constraints fail"

        beta_i_1 = 1
        Y_i_1 = np.random.binomial(1, beta_i_1, (batch_size, self.n))
        correlated_output = np.empty((batch_size, self.t, self.n), dtype=np.float32)
        correlated_output[:, 0] = np.random.binomial(
            1, activation_ratios[0], (batch_size, self.n)
        )
        for i in range(1, self.t - 1):
            p_i_1 = activation_ratios[i - 1]
            p_i = activation_ratios[i]
            rho_ = correlation_coefficients[i - 1]
            beta_i = np.sqrt(p_i_1 * p_i) / (
                np.sqrt(p_i_1 * p_i) + rho_ * np.sqrt((1.0 - p_i_1) * (1.0 - p_i))
            )
            alpha = p_i_1 / (beta_i * beta_i_1)
            assert (
                alpha <= 1
            ), "input parameters are not valid for generating correlated sequence"
            U = np.random.binomial(1, alpha, (batch_size, self.n))
            Y_i = np.random.binomial(1, beta_i, (batch_size, self.n))
            correlated_output[:, i] = U * Y_i * Y_i_1
            Y_i_1 = Y_i
            beta_i_1 = beta_i
        alpha = np.sqrt(activation_ratios[-1] / beta_i)
        beta_i = np.sqrt(activation_ratios[-1] / beta_i)
        U = np.random.binomial(1, alpha, (batch_size, self.n))
        Y_i = np.random.binomial(1, beta_i, (batch_size, self.n))
        correlated_output[:, -1] = U * Y_i * Y_i_1
        return correlated_output.astype(self.dtype)


class KDepedentCorrelation(BinaryCorrelatedData):
    """KDepedentCorrelation(t, n, dtype=np.float32)

    Generate binary data that are temporally correlated.

    This class randomly generates binary bits that are correlated over the given ``t`` time indices
    and the `n` number of bits.

    Parameters
    ----------
        t: int
            Defining the number of time indices that the bits are correlated at.

        n: int
            Defining the number of binary variables.

        k: int
            Defining the K-dependent length

        dtype: np.DType
            Defaults to `np.float32`. Defines the output datatype.

    Input
    -----
        [..., t], np.float32
            Activation ratios for Bernoulli random variables
        [..., t, t], np.float32
            Diagonal elements of correlation coefficient vectors for the `t` variables


    Output
    ------
        : [t, n], np.float32
            2D array containing the correlated binary bits with a shape of [t, n].

    Raises
    ------
        ValueError
            If inputs lengths are greater than `k`.


    Note
    ----
    Correlation is assumed to be existed in the time-domain.
    In the spatial domain, binary bits are independent.

    .. math::
            ┌ 1         r_{1,2}         r_{1,3}      ...    0 ┐
        R = │ r_{1,2}    1              r_{2,3}      ...    0 │
            │ ...       ...             ...          ...    : │
            └ 0          0              ...          ...    1 ┘

    Ref:
    * Wei Jiang, Shuang Song, Lin Hou & Hongyu Zhao (2021)
    A Set of Efficient Methods to Generate High-Dimensional Binary Data With Specified Correlation Structures,
    The American Statistician, 75:3, 310-322,
    """

    def __init__(self, t, n, k, dtype=np.float32):
        self._k = k
        super().__init__(t, n, dtype=dtype)

    @property
    def k(self):
        return self._k

    @staticmethod
    def generate_correlation_matrix(correlation_coefficients):
        """
        input is a list of diagonal correlation coefficients,
        diagonal elements of a correlation matrix are set to 0s
        """
        assert (
            len(correlation_coefficients) > 0
        ), "List size of correlation coefficients should be greater than 1"
        num_variables = len(correlation_coefficients[0]) + 1
        correlation_matrix = np.zeros((num_variables, num_variables), dtype=np.float32)
        for i in range(1, len(correlation_coefficients) + 1):
            if i < len(correlation_coefficients):
                if len(correlation_coefficients[i - 1]) < len(
                    correlation_coefficients[i]
                ):
                    raise ValueError("List should follow a descending order")
            ## upper triangle part
            diagonal_indices = np.arange(len(correlation_coefficients[i - 1]))
            correlation_matrix[
                diagonal_indices, diagonal_indices + i
            ] = correlation_coefficients[i - 1]
            ## lower triangle part
            diagonal_indices = np.arange(i, num_variables)
            correlation_matrix[
                diagonal_indices, diagonal_indices - i
            ] = correlation_coefficients[i - 1]
        return correlation_matrix

    def prentice_constrains(self, activation_ratios, correlation_coefficients_matrix):
        """Assume that correlation_coefficients is 1 dependent"""
        row_shape = correlation_coefficients_matrix.shape[0]
        column_shape = correlation_coefficients_matrix.shape[1]
        assert (
            row_shape == column_shape
        ), "Correlation coefficient matrix should be a square matrix"
        for i in range(row_shape):
            for j in range(i, column_shape):
                p_i = activation_ratios[i]
                p_j = activation_ratios[j]
                coefficient = correlation_coefficients_matrix[i, j]
                left_factor = np.sqrt(p_i * (1.0 - p_j) / (p_j * (1.0 - p_i)))
                right_factor = np.sqrt(p_j * (1.0 - p_i) / (p_i * (1.0 - p_j)))
                if max(-left_factor, -right_factor) > coefficient:
                    return False
                elif min(left_factor, right_factor) < coefficient:
                    return False
        return True

    def __call__(self, activation_ratios, correlation_coefficients, batch_size=1):
        assert isinstance(
            activation_ratios, np.ndarray
        ), "inputs should be numpy.ndarray"
        assert (
            activation_ratios.shape[-1] == self.t
        ), "activation ratio shape does not match preset"
        assert isinstance(
            correlation_coefficients, np.ndarray
        ), "correlation coefficients should be 2D numpy.ndarray"
        assert self.prentice_constrains(
            activation_ratios, correlation_coefficients
        ), "Prentice constraints fail"

        correlated_output = np.empty((batch_size, self.t, self.n), dtype=np.float32)
        redundant_activation_ratios = activation_ratios[-1] * np.ones(
            (self.k), dtype=np.float32
        )
        activation_ratios = np.concatenate(
            [activation_ratios, redundant_activation_ratios], axis=0
        )
        Y = np.empty((batch_size, self.k, self.t, self.n), dtype=np.float32)
        beta = np.empty((self.k, self.t), dtype=np.float32)
        for i in range(self.k):
            for j in range(self.t):
                p_j = activation_ratios[j]
                p_i_j = activation_ratios[j + i + 1]
                if i + j + 1 >= self.t:
                    rho = 0.0
                else:
                    rho = correlation_coefficients[j, i + j + 1]
                beta[i, j] = (
                    p_j
                    * p_i_j
                    / (
                        p_j * p_i_j
                        + rho * np.sqrt(p_j * p_i_j * (1.0 - p_j) * (1.0 - p_i_j))
                    )
                )
                Y[:, i, j] = np.random.binomial(1, beta[i, j], (batch_size, self.n))

        alpha = activation_ratios[0] / (np.prod(beta[: self.k, 0], axis=0))
        assert alpha < 1, "Unavailable correlation structure"
        U = np.random.binomial(1, alpha, self.n)

        correlated_output[:, 0] = U * np.prod(Y[:, 0], axis=1)
        for i in range(1, self.t):
            K_ = min(self.k, i - 1)
            alpha = activation_ratios[i] / (
                np.prod(beta[: self.k, i], axis=0)
                * np.prod(beta[np.arange(K_), i - np.arange(K_)], axis=0)
            )
            assert alpha < 1, "Unavailable correlation structure"
            U = np.random.binomial(1, alpha, (batch_size, self.n))
            correlated_output[:, i] = (
                U
                * np.prod(Y[:, : self.k, i], axis=1)
                * np.prod(Y[:, np.arange(K_), i - np.arange(1, K_ + 1)], axis=1)
            )
        return correlated_output.astype(self.dtype)


class DataGenerator:
    """
    Choose the binary correlated model and generate binary bits


    Parameters
    ----------
        correlated_time_duration: int
            Defining the length of time indices across time domain

        sequence_length: int
            Defining the number of samples in the space domain

        activation_ratios: numpy.array
            Defining the binary variables' activation ratios across time domain

        correlation_coefficients: numpy.array
            Defining the correlated factors (size will depend on the correlation structure)

        memory_buffer: int
            Defining the size of memory buffer to generate features

        k_dependent: int
            Defining the number of correlated binary variables

        type_name: str
            Choose the type of correlation structures ("K-dependent", "One-dependent" or "Decaying Product")


    Input
    -----
        : batch_size int
            batch size of the generated random binary sequences


    Output
    ------
        : [batch, t, n], np.float32
            3D array containing the correlated binary bits with a shape of [batch, t, n].


    Note
    ----
    The activation ratios and correlation coefficients need to match the preset requirements for each type of
    correlated structures
    """

    def __init__(
        self,
        correlated_time_duration,
        sequence_length,
        activation_ratios,
        correlation_coefficients,
        memory_buffer=3,
        k_dependent=2,
        type_name="OneDependent",
    ):
        assert activation_ratios.size == correlated_time_duration
        self._sequence_length = sequence_length
        self._activation_ratios = activation_ratios
        self._correlation_coefficients = correlation_coefficients
        self._correlated_time_duration = correlated_time_duration
        self._type_name = type_name
        self._memory_buffer = min(memory_buffer, self.correlated_time_duration - 1)
        if type_name == "KDependent":
            self.data_model = KDepedentCorrelation(
                self.correlated_time_duration, self.sequence_length, k_dependent
            )

        elif type_name == "OneDependent":
            self.data_model = OneDepedentCorrelation(
                self.correlated_time_duration, self.sequence_length
            )

        elif type_name == "DecayingProduct":
            self.data_model = DecayingProductCorrelation(
                self.correlated_time_duration, self.sequence_length
            )

        else:
            self.generator = self.random_binary

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def activation_ratios(self):
        return self._activation_ratios

    @property
    def correlation_coefficients(self):
        return self._correlation_coefficients

    @property
    def type_name(self):
        return self._type_name

    @property
    def correlated_time_duration(self):
        return self._correlated_time_duration

    @property
    def memory_buffer(self):
        return self._memory_buffer

    def random_binary(
        self, batch_size=1, activation_ratios=None, correlation_coefficients=None
    ):
        """random binary sequence with preset activation ratio"""
        data = np.empty(batch_size, self.correlated_time_duration, self.sequence_length)
        for i in range(self.correlated_time_duration):
            data[:, i, :] = np.random.binomial(
                1,
                self.activation_ratios[i],
                (batch_size, self.sequence_length),
            )
        return data

    def __call__(self, batch_size=1):
        """data shape: (batch, num_TTIs, feature length)"""
        return self.data_model(
            self.activation_ratios, self.correlation_coefficients, batch_size=batch_size
        )
