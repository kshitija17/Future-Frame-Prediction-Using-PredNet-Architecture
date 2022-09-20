import hickle as hkl
import os
import numpy as np


class DataLoad():
    def __init__(self):
        pass

    def __getitem__(self, i):
        loc = self.snippet_starts[i]
        return self.x[loc:loc + time_steps]

    def __len__(self):
        return len(self.snippet_starts)

    def __call__(self, dataset_file, dataset_info, time_steps):
        self.dataset_file = dataset_file
        self.dataset_info = dataset_info
        self.x = hkl.load(self.dataset_file)
        self.info = hkl.load(self.dataset_info)

        self.time_steps = time_steps

        curr_location = 0
        snippets = []

        while curr_location < self.x.shape[0] - self.time_steps + 1:
            if self.info[curr_location] == self.info[curr_location + time_steps - 1]:
                one_snippet = self.x[curr_location:curr_location + time_steps]

                snippets.append(one_snippet)

                curr_location += time_steps
            else:
                curr_location += 1

        self.snippets = snippets
        snippets_list = np.array(self.snippets).astype(np.float32)
        snippets_list = snippets_list / 255

        return snippets_list

    def _create_batch(self, arr, batch_size):

        length = arr.shape[0]
        start_index = 0
        end_index = batch_size
        arr_batches = []

        while end_index <= length:
            arr_batches.append(arr[start_index: end_index])
            start_index += batch_size
            end_index = start_index + batch_size

        return arr_batches