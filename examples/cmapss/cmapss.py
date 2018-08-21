import numpy as np
import torch
import torch.utils.data as data


class CMAPSSData(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        mean = [1.72245631e+01, 4.10352811e-01, 9.57339325e+01,
                4.85821285e+02, 5.97278898e+02, 1.46612718e+03,
                1.25948690e+03, 9.89215430e+00, 1.44201815e+01,
                3.59590653e+02, 2.27384447e+03, 8.67514621e+03,
                1.15343657e+00, 4.41669423e+01, 3.38650301e+02,
                2.34971089e+03, 8.08726502e+03, 9.05152236e+00,
                2.51273110e-02, 3.60457294e+02, 2.27378874e+03,
                9.83927591e+01, 2.59455399e+01, 1.55675691e+01]
        stddev = [1.65287845e+01, 3.67992288e-01, 1.23467959e+01,
                  3.04228203e+01, 4.24595469e+01, 1.18054339e+02,
                  1.36086086e+02, 4.26553251e+00, 6.44366063e+00,
                  1.74102867e+02, 1.42310999e+02, 3.74093117e+02,
                  1.42073445e-01, 3.41967503e+00, 1.64160057e+02,
                  1.11057510e+02, 7.99949844e+01, 7.50328573e-01,
                  4.99837893e-03, 3.09876001e+01, 1.42395993e+02,
                  4.65165534e+00, 1.16951710e+01, 7.01722544e+00]

        def read_data(format_str, index):
            data = np.loadtxt(format_str.format(index))
            data[:, 0] = data[:, 0] + 1000 * index
            return data

        if self.train:
            data = np.concatenate(
                [read_data('data/cmapss/train_FD00{}.txt', i)
                 for i in [1, 2, 3, 4]]
            )
        else:
            data = np.concatenate(
                [read_data('data/cmapss/test_FD00{}.txt', i)
                 for i in [1, 2, 3, 4]]
            )
            self.y = np.concatenate(
                [np.loadtxt(f'data/cmapss/RUL_FD00{i}.txt')
                 for i in [1, 2, 3, 4]]
            )

        # Normalize data
        data[:, 2:26] = (data[:, 2:26] - mean) / stddev

        # Split data into each engines timesteps
        samples_per_engine = np.unique(data[:, 0], return_counts=True)[1]
        start_index_per_engine = np.cumsum(samples_per_engine)[:-1]
        engine_samples = np.split(data, start_index_per_engine, axis=0)

        self.samples = engine_samples

    def __getitem__(self, index):
        sample = self.samples[index]
        max_day = sample[-1, 1]
        features = sample[:np.random.randint(sample.shape[0] - 5,
                                             sample.shape[0] - 1),
                          2:26]
        seq_len = len(features)
        inputs = torch.FloatTensor(features)
        tte = int(max_day + (0 if self.train else self.y[index]))
        targets = torch.arange(tte, tte - seq_len, -1)
        targets = torch.stack((targets, torch.ones(targets.size())), 1)

        return inputs, targets

    def __len__(self):
        return len(self.samples)
