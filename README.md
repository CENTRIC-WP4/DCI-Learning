# Downlink Control Information (DCI) Learning
Author: Bryan Liu - Nokia Network France

This repository generates random correlated binary sequences that might be used to form a DCI message. 
The encoding and decoding steps follow 5G New Radio 3GPP standards.
This repository might be used to test compression methods for customized DCI messages.


## Overview
There are 4 types of methods for generating (un)correlated random control bits.
* **KDependent**: adjacent K bits are correlated following a preset vector of correlation coefficient.
* **OneDependent**: adjacent 2 bits are correlated following a preset correlation coefficient.
* **DecayingProduct**: the closer the indices of the bits, the higher the correlation between the bits.
* **Random**: binary uncorrelated bits are generated

Besides generating the (un)correlated binary sequences, there is a built-in non-contiguous even resource scheduler, 
which allocates equal number of resources to the User Equipments (UEs).
The scheduling decisions are mapped to a DCI message together with the decision of Modulation and Coding Scheme (MCS).

## Expected outputs
Default main.py generates two dictionaries:
* DCI messages: with User Equipment Id as the keys and DCI payloads as the values. 
DCI payload is a 2D array with a shape of (number of TTIs, DCI payload length)
* Encoded DCI messages: with User Equipment Id as the keys and encoded DCIs as the values.
Encoded DCI message is a 2D array with a shape of (number of TTIs, encoded DCI length)

Corresponding compressed DCI messages for the test dataset with baseline algorithms, Huffman Coding or Lempel-Ziv-Welch, are generated.


## Getting started
1. Get the code:
    ```
    git clone https://github.com/CENTRIC-WP4/DCI-Learning
    ```

2. Use `pip3` to install the package:
   ```
   pip3 install -r requirements.txt
   ```
   
3. Run the code:
   ```
   python main.py
   ```

## Baselines
Two baselines are provided for lossless data compression.
* [Lempel-Ziv-Welch](https://github.com/stensaethf/Lempel-Ziv-Welch-Compressor)
* [Huffman Coding](https://github.com/bhrigu123/huffman-coding)

## Usage
The following parameters can be configured for generalization test, 
where the control bits for frequency domain scheduling and MCS decisions can be generated from a different distribution.
* num_ues: number of UEs in the network.
* poisson_rate: average arrival rate for the traffic model, which follows a Poisson distribution.
* num_RBs: number of available resource blocks.
* max_buffer_size: maximum buffer size.


## How to contribute
There are two main ways of contributing to this repository

1. **Implementing new problems**: 
Under various system models, the DCI messages should follow different patterns. 
E.g., introducing a different channel model and traffic model would return DCI messages with different distributions.
There are multiple fields in a DCI message that can be added to the repository, 
e.g., introducing a closed-loop power control command is included in a DCI message
2. **Implementing new solutions**: 
Lossless data compression is an on-going research topic. 
In particular, with the assistance of machine learning, the distribution of source data symbols might be better estimated to achieve a higher compression ratio

## References
* [5G-NR Polar Coding](https://github.com/vodafone-chair/5g-nr-polar/tree/master)
* [Sionna](https://nvlabs.github.io/sionna/)
* [Reporting of CQI](https://uk.mathworks.com/help/lte/ug/reporting-of-channel-quality-indicator-cqi-conformance-test.html)
* [Lempel-Ziv-Welch](https://archive.wikiwix.com/cache/index2.php?url=http%3A%2F%2Fwww.csa.com%2Fpartners%2Fviewrecord.php%3Fcollection%3DTRD%26recid%3DA8436773AH#federation=archive.wikiwix.com&tab=url)
* [Huffman Coding](http://compression.ru/download/articles/huff/huffman_1952_minimum-redundancy-codes.pdf)

## License
This project is licensed under the BSD-3-Clause license - see the [LICENSE](https://github.com/CENTRIC-WP4/DCI-Learning?tab=BSD-3-Clause-1-ov-file#readme)