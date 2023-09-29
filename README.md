# Downlink Control Information (DCI) Learning
Author: Bryan Liu - Nokia Network France

This repository generates random correlated binary sequences that might be used to form a DCI message. 
The encoding and decoding steps follow 5G New Radio 3GPP standards.

Most of the codes are written based on the following resources:
* https://github.com/vodafone-chair/5g-nr-polar/tree/master
* https://nvlabs.github.io/sionna/
* https://uk.mathworks.com/help/lte/ug/reporting-of-channel-quality-indicator-cqi-conformance-test.html

This repository might be used to test compression methods for DCI messages.

```commandline
python main.py
```
Default main.py generates two dictionaries:
* DCI messages: with User Equipment Id as the keys and DCI payloads as the values. 
DCI payload is a 2D array with a shape of (number of TTIs, DCI payload length)
* Encoded DCI messages: with User Equipment Id as the keys and encoded DCIs as the values.
Encoded DCI message is a 2D array with a shape of (number of TTIs, encoded DCI length)

### Folders
* **pdcch**: following the 5GNR standards, a DCI message in this code repository is encoded by the steps of 
  * CRC attachment
  * CRC scramling
  * Interleaving
  * Polar encoding
  * Rate matching
* **control_bits**: generate DCI control bits. Besides the scheduling decision and MCS indices 
that are generated according to a pre-defined channel model, other control bits are assumed 
to be generated from a correlated structure that can either be "KDependent", "OneDependent" or "DecayingProduct".