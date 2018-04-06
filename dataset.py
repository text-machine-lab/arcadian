from abc import ABC, abstractmethod
import numpy as np
import random

class Dataset(ABC):
    """Input to all GenericModel subclasses. Overload to implement a variety of datasets.
    For a dataset to apply, must be able to index into dataset to return dictionary of features
    for a single example. Must implement __len__ function to specify size of dataset. Complete
    processing of dataset is encouraged to occur within a Dataset object. If retrieving single
    examples during training is too slow, the generate_batches() function can be overridden."""

    @abstractmethod
    def __getitem__(self, index):
        # Get a single item as an index from the dataset.
        # {'feature1': np_array1, 'feature2': np_array2}
        # {'history': np_array1, 'response': np_array2}
        pass

    @abstractmethod
    def __len__(self):
        # Return the length of the dataset.
        pass

    def generate_batches(self, batch_size, shuffle=False):
        # Yield one batch from the dataset
        m = len(self)

        # We want the ability to have an empty dataset,
        # for output tensors which require no input
        if m == 0:
            return

        indices = np.arange(m)
        if shuffle:
            np.random.shuffle(indices)
        num_batches = m // batch_size + 1

        for i in range(num_batches):
            index_batch = indices[i * batch_size:i * batch_size+batch_size]
            if len(index_batch) == 0:
                break

            batch_data = [self[each_index] for each_index in index_batch]
            #
            # if len(batch_data) > 32:
            #     print('batch_data > 32!')
            #     print('batch_size: %s' % batch_size)
            #     print('num_batches: %s' % num_batches)
            #     print('Dataset len: %s' % len(self))
            #     print('Num indices: %s' % len(indices))

            result = {}

            assert batch_data is not None
            assert batch_data[0] is not None

            for key in batch_data[0]:
                result[key] = np.stack([d[key] for d in batch_data], axis=0)

            yield concatenate_batch_dictionaries(batch_data, single_examples=True)


    def split(self, fraction, seed='seed', max_examples=None):
        """Split dataset into two subset datasets. 'fraction' argument decides
        what fraction of the original dataset is used to make the first subset.
        The remaining examples are used to create the second subset. Typically
        used for train/test splits. By DEFAULT there is a seed, as randomly
        splitting training and test sets on each run can leave training examples
        in the validation set and ultimately a higher validation accuracy.

        Arguments:
            - fraction: fraction of examples in first subset
            - seed: random seed for splitting dataset
            - max_examples: number of examples to use from dataset in split
        """

        m = len(self) if max_examples is None else max_examples
        dataset_indices = list(range(m))

        if seed is not None:
            random.seed(seed)

        random.shuffle(dataset_indices)
        subset_divider_index = int(m * fraction)
        first_subset = DatasetPtr(self, dataset_indices[:subset_divider_index])
        second_subset = DatasetPtr(self, dataset_indices[subset_divider_index:])
        return first_subset, second_subset

    def to_numpy(self, feature):
        """Grab all examples of feature and concatenate them into a numpy array.

        Returns: a numpy array"""
        examples = []
        for index in range(len(self)):
            examples.append(self[index][feature])

        return np.stack(examples, axis=0)



class RenameDataset(Dataset):
    def __init__(self, dataset, mappings):
        """Takes features from dataset and renames
        them.

        mappings - dictionary with keys as
        features in original dataset and values
        as the renamed versions"""
        self.dataset = dataset
        self.mappings = mappings

        # add remaining features back as pointing to themselves
        features = dataset[0].keys()
        for feature in features:
            if feature not in mappings:
                self.mappings[feature] = feature

    def __getitem__(self, index):
        return {self.mappings[feature]: self.dataset[index][feature] for feature in self.dataset[index]}

    def __len__(self):
        return len(self.dataset)


class DatasetPtr(Dataset):
    def __init__(self, dataset, indices):
        """Take as argument a dataset, and a list of indices
        into that dataset. Sample from those indices to create
        a subset dataset of the original dataset. Typically used
        for breaking a dataset into subsets."""
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        """Grab the index'th index into the dataset
        and return that data example."""
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class EmptyDataset(Dataset):
    """Dataset with no features,
    runs for only a single batch"""
    def __getitem__(self, index):
        return {}

    def __len__(self):
        return 1


class DictionaryDataset(Dataset):
    def __init__(self, batch_feature_dict):
        """Create simple dataset from a dictionary of feature_name:numpy_feature entries.
        Interpret as a series of features, each having a name the model can refer to.
        Optionally, define dictionary of scalar features which remain the same across
        examples. The first dimension of all features must be equal to the length
        of the dataset."""
        self.batch_feature_dict = batch_feature_dict

        # Determine length of dataset
        self.length = -1
        for feature_name in self.batch_feature_dict:
            feature_length = self.batch_feature_dict[feature_name].shape[0]
            if self.length == -1:
                self.length = feature_length
            else:
                if self.length != feature_length:
                    raise ValueError('All batch features must have same length')

        if self.length <= 0:
            raise ValueError('Cannot have zero-length dataset.')

    def __getitem__(self, index):
        item_dict = {}
        for feature_name in self.batch_feature_dict:
            item_dict[feature_name] = self.batch_feature_dict[feature_name][index]
        return item_dict

    def __len__(self):
        return self.length

    def to_dict(self):
        return self.batch_feature_dict

class MergeDataset(Dataset):
    def __init__(self, datasets, concat_duplicates=False):
        """Dataset intended to merge the features of multiple datasets.

        Arguments:
            - datasets: list of Dataset objects, all with the same length
        """
        self.datasets = datasets.copy()

        for index in range(len(self.datasets)):
            if isinstance(self.datasets[index], dict):
                self.datasets[index] = DictionaryDataset(self.datasets[index])

        # Get all keys
        self.keys = []
        for dataset in self.datasets:
            for feature in dataset[0]:
                if feature not in self.keys:
                    self.keys.append(feature)

        # Create mapping from keys (features) to datasets that contain them
        self.has_duplicates = False
        self.feat2data = {}
        for feature in self.keys:
            self.feat2data[feature] = [dataset for dataset in self.datasets if feature in dataset[0]]
            if len(self.feat2data[feature]) == 1:
                self.feat2data[feature] = self.feat2data[feature][0]  # only one item in list, remove list wrapper
            else:
                self.has_duplicates = True

        assert not self.has_duplicates or concat_duplicates

        self.length = len(self.datasets[0])
        for each_dataset in self.datasets:
            assert len(each_dataset) == self.length

    def __getitem__(self, index):
        output_dict = {}
        for feature in self.feat2data:
            data = self.feat2data[feature]  # datasets having this key (feature name)
            if isinstance(data, Dataset):
                output_dict[feature] = data[index][feature]
            else:
                # it is a list of datasets, we concatenate everything along the feature axis
                output_dict[feature] = np.concatenate([d[index][feature] for d in data], axis=0)

        return output_dict

    def __len__(self):
        return self.length

def concatenate_batch_dictionaries(batch_dictionaries, single_examples=False):
    """Concatenates numpy dictionaries. If numpy arrays represent single examples (no batch axis),
    set single_examples=True. Otherwise false.
    batch_dictionaries - list of dictionaries, all containing identical keys, each key being
    a feature name
    single_examples - decides whether to concatenate (for batches) or to stack (for single vectors)"""
    result = {}

    # If no features, return None
    # if len(batch_dictionaries) == 0:
    #     return None

    for key in batch_dictionaries[0]:
        if single_examples or len(batch_dictionaries[0][key].shape) == 0:
            tensors = [d[key] for d in batch_dictionaries]
            result[key] = np.stack(tensors, axis=0)
        else:
            result[key] = np.concatenate([d[key] for d in batch_dictionaries], axis=0)

    return result