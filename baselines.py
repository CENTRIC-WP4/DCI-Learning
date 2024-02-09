"""
This file contains lossless compression methods: Lempel-Ziv-Welch and Huffman Coding
Most of the functions can be found from the open Github repositories:
* Huffman coding: https://github.com/bhrigu123/huffman-coding
* Lempel-Ziv-Welch: https://github.com/stensaethf/Lempel-Ziv-Welch-Compressor
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd


def string2bits(s=""):
    """
    Convert a string of characters to a list of binary bits

    Input
    ----------
    s: str
    A string of characters

    Output
    -------
    A list of binary bits converted from the input string (character by character).
    """
    return [bin(ord(x))[2:].zfill(8) for x in s]


def bin_to_array(binString):
    """
    Convert a binary string to a numpy array, where each bit becomes an element in the array

    Input
    ----------
    binString: string
    A string that contains binary values (0, 1)

    Returns
    -------
    A numpy array that converts the binary string to a numpy array
    """
    return np.array([int(b) for b in binString], dtype=np.int32)


def convert_bit_char(bit_array):
    """
    Given a numpy array that contains binary values, convert to a character string
    Each character is represented by 8 bits. Therefore, zero-paddings are added if the
    length of binary array is not a multiple of 8

    Input
    ----------
    bit_array: numpy.array
    A numpy array that contains binary values

    Returns
    -------
    A character string that is converted from the binary sequence
    """
    bit_array = bit_array.reshape(-1).astype(int)
    bit_string = "".join([str(b) for b in bit_array])
    num_zero_padding = 8 - len(bit_string) % 8
    initial_text = bit_string + "0" * num_zero_padding
    char_string = (initial_text[i : i + 8] for i in range(0, len(initial_text), 8))
    return "".join(chr(int(char, 2)) for char in char_string)


class Huffman_coding:
    """
    Please refer to https://github.com/bhrigu123/huffman-coding for more details
    There are minor changes, where the input of the compression function is assumed to be a binary array
    """

    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if other == None:
                return False
            if not isinstance(other, HeapNode):
                return False
            return self.freq == other.freq

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root == None:
            return

        if root.char != None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i : i + 8]
            b.append(int(byte, 2))
        return b

    def update_dictionary(self, train_data):
        self.train_text = convert_bit_char(train_data)
        frequency = self.make_frequency_dict(self.train_text)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

    def compress(self, binary_array):
        ## convert binary input to bit string -> character string
        new_string = convert_bit_char(binary_array)
        ## reinitialize the dictionary
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        frequency = self.make_frequency_dict(self.train_text + new_string)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()
        encoded_text = self.get_encoded_text(new_string)
        padded_encoded_text = self.pad_encoded_text(encoded_text)
        b = self.get_byte_array(padded_encoded_text)
        return b

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[: -1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""
        return decoded_text

    def byte_to_bit_string(self, compressed_data):
        bit_string = ""
        for byte in compressed_data:
            bits = bin(byte)[2:].rjust(8, "0")
            bit_string += bits
        return bit_string

    def decompress(self, compressed_data):
        bit_string = self.byte_to_bit_string(compressed_data)
        encoded_text = self.remove_padding(bit_string)
        decompressed_text = self.decode_text(encoded_text)
        decompressed_text = "".join(string2bits(decompressed_text))
        assert len(decompressed_text) > 0, "Please check training and test data"
        return bin_to_array(decompressed_text)


class LZW:
    """
    Please refer to https://github.com/stensaethf/Lempel-Ziv-Welch-Compressor for more details
    """

    def __init__(self):
        self.reinitialize()

    def reinitialize(self):
        self.dict_size = 256
        self.dictionary = dict((chr(i), i) for i in range(self.dict_size))

    def int_to_binary(self, x):
        bits = self.num_bits
        return np.array([int(i) for i in bin(x)[2:].zfill(bits)])

    def binary_to_int(self, binary_sequence):
        if len(binary_sequence) == 0:
            raise ValueError("Binary sequence should not be empty")
        padding = len(binary_sequence) % self.num_bits
        num_segmenets = int(len(binary_sequence) / self.num_bits)
        num_segmenets = num_segmenets + 1 if padding > 0 else num_segmenets
        int_array = []
        for i in range(num_segmenets):
            if i == num_segmenets - 1:
                segment = np.hstack(
                    [np.zeros(padding, np.int32), binary_sequence[i * self.num_bits :]]
                )
            else:
                segment = binary_sequence[i * self.num_bits : (i + 1) * self.num_bits]
            int_value = sum(2 ** np.arange(self.num_bits) * segment[::-1])
            int_array.append(int_value)
        return int_array

    @property
    def num_bits(self):
        return int(np.ceil(np.log2(self.dict_size)))

    def update_dictionary(self, uncompressed):
        if uncompressed is None:
            return None
        uncompressed = convert_bit_char(uncompressed)
        w = ""
        for c in uncompressed:
            wc = w + c
            if wc in self.dictionary:
                w = wc
            else:
                self.dictionary[wc] = self.dict_size
                self.dict_size += 1
                w = c

    def compress(self, uncompressed):
        uncompressed = convert_bit_char(uncompressed)
        w = ""
        result = []
        for c in uncompressed:
            wc = w + c
            if wc in self.dictionary:
                w = wc
            else:
                result.append(self.dictionary[w])
                self.dictionary[wc] = self.dict_size
                self.dict_size += 1
                w = c
        if w:
            result.append(self.dictionary[w])

        results = np.concatenate([self.int_to_binary(r) for r in result], axis=0)
        return results

    def decompress(self, compressed_data):
        integer_array = self.binary_to_int(compressed_data)
        compressed_data = integer_array

        temp_table = {}
        for key, value in self.dictionary.items():
            temp_table[value] = key

        prev = temp_table[compressed_data[0]]
        compressed_data = compressed_data[1:]
        decompressed_str = prev
        for element in compressed_data:
            if element == len(temp_table):
                string = prev + prev
            elif element not in temp_table:
                raise ValueError("Invalid element in the compressed list")
            else:
                string = temp_table[element]
            decompressed_str += string
            temp_table[len(temp_table)] = prev + string[0]
            prev = string
        decompressed_str = "".join(string2bits(decompressed_str))
        return bin_to_array(decompressed_str)


def lossless_compression(method, train_x, test_x):
    """
    Perform lossless compression. Baseline algorithms: Huffman Coding and Lempel-Ziv-Welch

    Parameters
    ----------
    method: bool
        Method selection
    train_x: np.array
        Used for updating the dictionary / machine learning model
    test_x: np.array
        Data to be compressed

    Returns
    -------
    Losslessly compressed data
    """
    if method == "LZW":
        compressor = LZW()
    elif method == "HC":
        compressor = Huffman_coding()
    else:
        raise ValueError("Please select a valid compression method")

    ## FIXME: may consider reducing the number of training data for LZW
    compressor.update_dictionary(train_x)

    ## Huffman coding and LZW are dictionary based lossless compression methods, combine
    ## both train and test dataset to setup the dictionary
    decompressed = []
    compressed = []
    for x in test_x:
        c = compressor.compress(x)
        if method == "HC":
            binary_array = bin_to_array(compressor.byte_to_bit_string(c))
            compressed.append(binary_array)
        elif method == "LZW":
            compressed.append(c)
        else:
            pass

        ## assume that no compression is needed if the compressed data exceeds original length
        if len(compressed[-1]) > test_x.shape[-1]:
            compressed[-1] = x

        ## crop the decompressed sequence if the number of bits does not form a byte
        num_paddings = 8 - test_x.shape[-1] % 8
        decompressed.append(compressor.decompress(c)[:-num_paddings])
    assert (np.vstack(decompressed) == test_x).all(), "Compression failed"
    return compressed


def train_test_split(data_x, train_size=0.7, batch_first=True):
    """
    Split the data set into training and test data

    Inputs
    ----------
    data_x: numpy.array
        Data to be splitted into training and test sets
    split_ratio: float
        A ratio that splits the number of samples in the input dataset
    batch_first: bool
        Determine whether the initial dimension refers to the dimension of batch

    Returns
    -------
    train_x: numpy.array
        Training set that has a size of int(number of batches * train size ratio)
    test_x: numpy.array
        Test set that has a size of (number of batches) - int(number of batches * train size ratio)
    """
    if not batch_first:
        data_x = np.swapaxes(data_x, 0, 1)
    ## assume that all -1 rows are null
    data_x = data_x[np.sum(data_x, -1) != -data_x.shape[-1]]
    num_batch = data_x.shape[0]
    num_training_data = int(num_batch * train_size)
    num_test_data = num_batch - int(num_batch * train_size)
    return data_x[:num_training_data], data_x[-num_test_data:]


class CompressionStatisticCollector:
    def __init__(self, name=""):
        self.stats_tuple = namedtuple(
            "stats_tuple", ["idx", "original", "compressed", "compressed_length"]
        )
        self.stats = []
        self.plot_smooth_factor = 0.6
        self.name = name

    @property
    def stats_df(self):
        return pd.DataFrame(self.stats, columns=self.stats_tuple._fields)

    def add(self, timestamp, original, compressed, compressed_length):
        if compressed_length > 0:
            compressed = np.concatenate(
                [compressed, 2 * np.ones(compressed_length, dtype=np.int32)], axis=0
            )
        self.stats.append(
            self.stats_tuple(timestamp, original, compressed, compressed_length)
        )

    @property
    def average_compressed_length(self):
        return self.stats_df["compressed_length"].mean()

    def plot_heatmap(self, saver_directory):
        if len(self.stats_df.index) <= 1:
            return None

        original = np.concatenate([self.stats_df["original"].to_list()], axis=0)
        compressed = np.concatenate([self.stats_df["compressed"].to_list()], axis=0)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

        ax1.imshow(original.T, cmap="Blues", interpolation="nearest", vmin=0, vmax=2)
        ax2.imshow(compressed.T, cmap="Blues", interpolation="nearest", vmin=0, vmax=2)

        # Where we want the ticks, in pixel locations
        ticks = np.linspace(0, len(self.stats_df.index) - 1, 2).astype(int)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(self.stats_df["idx"].iloc[ticks])
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(self.stats_df["idx"].iloc[ticks])
        ax1.set_xlabel('DCI Index')
        ax1.set_ylabel('Source Data')
        ax2.set_xlabel('DCI Index')
        ax2.set_ylabel('Compressed Data')

        plt.tight_layout()
        plt.savefig(f"{saver_directory}/stats_plot_{self.name}.pdf", dpi=500)
        plt.close(fig)
