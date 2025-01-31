"""
This code follows the DeepZip architecture but with a simple DNN  as the neural estimator
Feel free to train your own model for better compression performance

PLease refer to https://github.com/mohit1997/DeepZip for more details

"""

from .arithmeticcoding_fast import (
    BitOutputStream,
    ArithmeticEncoder,
    BitInputStream,
    ArithmeticDecoder,
)
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd
import pickle


class MLStatisticCollector:
    def __init__(self):
        self.stats_tuple = namedtuple("stats_tuple", ["validation_loss", "batch_index"])
        self.stats = []

    @property
    def stats_df(self):
        return pd.DataFrame(self.stats, columns=self.stats_tuple._fields, dtype="float")

    def add(self, validation_score, batch_index):
        self.stats.append(self.stats_tuple(validation_score, batch_index))

    def plot(self, saver_directory):
        ax = self.stats_df.plot.line(x='batch_index', y='validation_loss')
        ax.figure.savefig(f'{saver_directory}/stats_plot.pdf')
        plt.close(ax.get_figure())


class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim=64):
        super(DNNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.L_out = nn.Linear(hidden_dim // 4, self.output_dim)

    def forward(self, x):
        assert x.ndim > 2, "Check the input shape for DNN"
        x = self.dnn(x)
        out = nn.Sigmoid()(self.L_out(x))
        return out


class DeepZipDNN:
    def __init__(
            self,
            data_generator,
            feature_length,
            saver_directory="./output",
            device_index=0,
            num_batches=500000,
            validation_batches=200,
            batch_size=64,
    ):
        ## deep learning
        self.device = torch.device(
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
        self.num_batches = num_batches
        self.validation_batches = validation_batches
        self.DNN_predictor = DNNModel(feature_length, 1, self.device).to(self.device)
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.DNN_predictor.parameters(), lr=1e-4)
        self.stats_collector = MLStatisticCollector()
        self.saver_directory = saver_directory

        ## Arithmetic coding
        self.alphabet_size = 2
        self.stream_compression = BitOutputStream()
        self.encoder = ArithmeticEncoder(128, self.stream_compression)

    def train(self, saver_directory):
        """data_generator needs to contain a call function that returns a batch of training data"""
        self.validation_data, self.validation_label = self.data_generator(
            self.batch_size, validation=True
        )
        self.validation_data = torch.tensor(
            self.validation_data,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.validation_label = torch.tensor(
            self.validation_label,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        validation_best_score = self.loss(
            self.DNN_predictor(self.validation_data).squeeze(), self.validation_label
        )

        for b in tqdm(range(self.num_batches)):
            train_data, train_label = self.data_generator(self.batch_size)

            train_data = torch.tensor(
                train_data, device=self.device, dtype=torch.float32, requires_grad=False
            )
            train_label = torch.tensor(
                train_label,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )
            loss = self.loss(self.DNN_predictor(train_data).squeeze(), train_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if b % 300 == 0:
                self.validation_loss = self.loss(
                    self.DNN_predictor(self.validation_data).squeeze(), self.validation_label
                )
                print(f"Validation loss: {self.validation_loss.detach().numpy()} \n")
                self.stats_collector.add(self.validation_loss.detach().numpy(), batch_index=b)
                self.stats_collector.plot(self.saver_directory)
                if self.validation_loss < validation_best_score:
                    validation_best_score = self.validation_loss
                    self.save_models(saver_directory)

    @property
    def validation_loss(self):
        return self._value

    # setting the values
    @validation_loss.setter
    def validation_loss(self, value):
        self._value = value

    def compress(self, data, window_length=5):
        """data should be a [..., time, features] array"""
        num_sequence, codelength = data.shape
        data = data.squeeze()
        compressed_data = []
        TD_features = -1 * np.ones((window_length * codelength), dtype=np.float32)
        for i in range(num_sequence):
            self.encoder.renew()
            SD_features = -1 * np.ones((codelength - 1), dtype=np.float32)
            ## sequentially encode the control message
            for j, bit in enumerate(data[i]):
                feature_numpy = np.concatenate([TD_features, SD_features], axis=-1)
                feature = torch.tensor(
                    feature_numpy,
                    dtype=torch.float32,
                    device=self.device,
                )[None, None, ...]
                prob = self.DNN_predictor(feature).detach().numpy()
                prob = prob.squeeze()
                prob = np.array([1.0 - prob, prob])
                bit = np.int64(bit)
                cumul = np.zeros(self.alphabet_size + 1, dtype=np.uint64)
                cumul[1:] = np.cumsum(prob * 100000 + 1)
                self.encoder.write(cumul, bit)
                SD_features = np.roll(SD_features, -1)
                SD_features[-1] = bit
            TD_features = np.roll(TD_features, -codelength)
            TD_features[-codelength:] = np.array(data[i])
            self.encoder.finish()
            encoded = self.stream_compression.close()
            if encoded.shape[0] > data[i].shape[0]:
                encoded = data[i]
            compressed_data.append(encoded)
        return compressed_data

    def decompress(self, compressed, decoded_length, window_length):
        decoded_batch = []
        TD_features = -1 * np.ones((window_length * decoded_length), dtype=np.float32)
        ## assuming that the bits are compressed sequentially
        for i, c in enumerate(compressed):
            decoded_seq = []
            bitin = BitInputStream(c)
            dec = ArithmeticDecoder(128, bitin)
            SD_features = -1 * np.ones((decoded_length - 1), dtype=np.float32)
            for _ in range(decoded_length):
                feature_numpy = np.concatenate([TD_features, SD_features], axis=-1)
                feature = torch.tensor(
                    feature_numpy,
                    dtype=torch.float32,
                    device=self.device,
                )[None, None, ...]
                prob = self.DNN_predictor(feature).detach().numpy()
                prob = np.squeeze(prob)
                prob = np.array([1.0 - prob, prob])
                cumul = np.zeros(self.alphabet_size + 1, dtype=np.uint64)
                cumul[1:] = np.cumsum(prob * 100000 + 1)
                decoded_bit = dec.read(cumul, self.alphabet_size)

                SD_features = np.roll(SD_features, -1)
                SD_features[-1] = np.int64(decoded_bit)
                decoded_seq.append(decoded_bit)
            TD_features = np.roll(TD_features, -decoded_length)
            if c.shape[0] == decoded_length:
                decoded_seq = c
            TD_features[-decoded_length:] = np.array(decoded_seq)
            decoded_batch.append(np.array(decoded_seq))
        bitin.close()
        return decoded_batch

    def save_models(self, directory):
        torch.save(self.DNN_predictor.state_dict(), f"{directory}/DNN_model.pth")

    def load_models(self, directory):
        self.DNN_predictor.load_state_dict(
            torch.load(
                f"{directory}/DNN_model.pth",
                map_location=lambda storage, loc: storage,
            )
        )


# Function to save the object
def save_object(obj, filename="object.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


# Function to load the object
def load_object(filename="object.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


class DataGenerator:
    def __init__(
            self, dataset, memory_buffer=5,
    ):
        self._memory_buffer = memory_buffer
        self._dataset = dataset
        if self._dataset is not None:
            payload_length = self._dataset[0].shape[-1]
            self.feature_length = self._memory_buffer * payload_length + payload_length - 1
            self.prepare_training_dataset()

    @property
    def memory_buffer(self):
        return self._memory_buffer

    def prepare_training_dataset(self):
        self._train_valid_ratio = 0.8
        train_indices = [np.arange(x.shape[0])[
                         : int(x.shape[0] * self._train_valid_ratio)
                         ] for x in self._dataset]
        test_indices = [np.arange(x.shape[0])[
                        int(x.shape[0] * self._train_valid_ratio):
                        ] for x in self._dataset]
        self._train_set = [x[train_indices[i]] for i, x in enumerate(self._dataset)]
        self._valid_set = [x[test_indices[i]] for i, x in enumerate(self._dataset)]

        self.train_features_labels = self.combine_features_labels(
            [self.get_DNN_features(x[None, :]) for x in self._train_set])
        self.test_features_labels = self.combine_features_labels(
            [self.get_DNN_features(x[None, :]) for x in self._valid_set])

    def combine_features_labels(self, x_in):
        features = None
        label = None
        for x in x_in:
            features = np.vstack([features, x[0]]) if features is not None else x[0]
            label = np.vstack([label, x[1]]) if label is not None else x[1]
        return features, label

    def __call__(self, batch_size=32, validation=False):
        ## first dimension refers to the batch for multiple independent set of data
        if validation:
            min_num_samples = min([x.shape[0] for x in self._valid_set])
            possible_batch_ = min(min_num_samples, batch_size)
            indices = np.arange(min_num_samples)
            np.random.shuffle(indices)
            indices = indices[:possible_batch_]
            return self.test_features_labels[0][indices], self.test_features_labels[1][indices]
        else:
            ## train set might be a list of DCIs from different UE
            num_samples = self.train_features_labels[0].shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            possible_batch_ = min(num_samples, batch_size)
            indices = indices[:possible_batch_]
            return self.train_features_labels[0][indices], self.train_features_labels[1][indices]

    def get_TD_features(self, train_data):
        ## (batch * t, sequence_length, features) generate features from temporal and spatial information
        (batch_size, time_indices, sequence_length) = train_data.shape
        paddings_temporal = -1 * np.ones(
            (batch_size, self.memory_buffer, sequence_length), dtype=np.float32
        )
        data_padding_temporal = np.concatenate([paddings_temporal, train_data], axis=1)
        temporal_information = np.empty(
            (batch_size, time_indices, sequence_length * self.memory_buffer,),
            dtype=np.float32,
        )
        for i in range(self.memory_buffer):
            temporal_information[
            ..., i * sequence_length: (i + 1) * sequence_length
            ] = data_padding_temporal[:, i: time_indices + i]

        ## temporal information would be the same for each control bit in the spatial domain
        temporal_information = np.repeat(
            temporal_information, (sequence_length), axis=1
        )
        temporal_information = temporal_information.reshape(
            (
                batch_size * time_indices,
                sequence_length,
                sequence_length * self.memory_buffer,
            )
        )
        return temporal_information

    def get_SD_features(self, train_data):
        (batch_size, time_indices, sequence_length) = train_data.shape
        paddings_spatial = -1 * np.ones(
            (batch_size, time_indices, sequence_length - 1,), dtype=np.float32,
        )

        ## only the first bit has -1
        spatial_information = np.concatenate(
            [paddings_spatial, train_data[..., :-1]], axis=2
        )
        spatial_information_concat = []
        for i in range(sequence_length):
            spatial_information_concat.append(spatial_information[..., i:i + sequence_length - 1][..., None, :])

        spatial_information_concat = np.concatenate(spatial_information_concat, axis=-2)
        spatial_information_concat = spatial_information_concat.reshape(
            (
                batch_size * time_indices,
                sequence_length,
                sequence_length - 1,
            )
        )
        return spatial_information_concat

    def get_DNN_features(self, train_data):
        """data has a shape of (batch, k, n)"""
        (batch_size, time_indices, sequence_length) = train_data.shape

        ## original shape: (batch, t, n)
        label = train_data.reshape(batch_size * time_indices, sequence_length)

        temporal_information = self.get_TD_features(train_data)
        spatial_information_concat = self.get_SD_features(train_data)

        features = np.concatenate([temporal_information, spatial_information_concat], axis=-1, dtype=np.float32)
        return features, label
