import unittest2
import numpy as np
from arcadian.examples import SimpleModel, LessSimpleModel, EvenLessSimpleModel
import tensorflow as tf

from arcadian.dataset import DictionaryDataset, MergeDataset

class GenericModelTest(unittest2.TestCase):
    def test_merge_dataset(self):
        dataset1 = DictionaryDataset({'x': np.zeros([4, 5])})
        dataset2 = DictionaryDataset({'y': np.ones([4, 5])})
        merged_dataset = MergeDataset([dataset1, dataset2])
        assert len(merged_dataset) == len(dataset1)
        for index in range(len(dataset2)):
            example1 = dataset1[index]
            example2 = dataset2[index]
            example = merged_dataset[index]
            assert np.array_equal(example['x'], example1['x'])
            assert np.array_equal(example['y'], example2['y'])

    def test_batch_generator_and_dictionary_dataset_arange(self):
        """non-shuffled batches correspond to input feature dictionary. Test batch size."""
        data = np.random.uniform(size=(50, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=False)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert batch_dict[feature].shape[0] == batch_size
                assert np.array_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size])

    def test_batch_generator_and_dictionary_dataset_shuffle(self):
        """Set shuffle to true and batches will not correspond to input feature dictionary"""
        data = np.random.uniform(size=(50, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=True)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert np.not_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size]).any()

    def test_batch_generator_and_dictionary_dataset_arange_feature_vectors(self):
        """Test that DictionaryDataset batches correspond to input dataset."""
        data = np.random.uniform(size=(50, 5))
        feature_dict = {'f1': data[:, 0:2], 'f2': data[:, 2:4], 'f3': data[:, 4]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=False)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert np.array_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size])

    def test_batch_generator_and_dictionary_dataset_remainder(self):
        """Test that batch features have correct shape and DD can handle remainders."""
        data = np.random.uniform(size=(10, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 20
        for batch_dict in dts.generate_batches(batch_size=batch_size, shuffle=False):
            assert batch_dict['f1'].shape == batch_dict['f2'].shape
            assert batch_dict['f2'].shape == batch_dict['f3'].shape
            assert batch_dict['f1'].shape[0] == 10

    def test_batch_generator_and_dictionary_dataset_batch_size(self):
        """Test that batch features have correct shape and DD can handle correct batch size."""
        data = np.random.uniform(size=(100, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 20
        for batch_dict in dts.generate_batches(batch_size=batch_size, shuffle=False):
            assert batch_dict['f1'].shape == batch_dict['f2'].shape
            assert batch_dict['f2'].shape == batch_dict['f3'].shape
            assert batch_dict['f1'].shape[0] == 20

    def test_batch_generator_and_empty_dictionary_dataset(self):
        """You cannot create an empty dataset."""
        with self.assertRaises(ValueError):
            DictionaryDataset({})

    def test_dictionary_dataset(self):
        """Test that DictionaryDataset can handle multiple features."""
        data = np.random.uniform(size=(10, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        for feature_name in feature_dict:
            feature = feature_dict[feature_name]
            for index in range(feature.shape[0]):
                example_value = feature[index]
                dataset_example = dts[index]
                assert isinstance(dataset_example, dict)
                #print(dataset_example)
                dataset_example_value = dataset_example[feature_name]
                assert np.array_equal(example_value, dataset_example_value)

    def test_dictionary_dataset_vector_features(self):
        """Test that the DictionaryDataset can handle multi-dimensional features."""
        data = np.random.uniform(size=(10, 5))
        feature_dict = {'f1': data[:, 0:2], 'f2': data[:, 2:4], 'f3': data[:, 4]}
        dts = DictionaryDataset(feature_dict)
        for feature_name in feature_dict:
            feature = feature_dict[feature_name]
            for index in range(feature.shape[0]):
                example_value = feature[index]
                dataset_example = dts[index]
                assert isinstance(dataset_example, dict)
                #print(dataset_example)
                dataset_example_value = dataset_example[feature_name]
                assert np.array_equal(example_value, dataset_example_value)

    def test_simple_model_creation(self):
        """Test that SimpleModel can be instantiated."""
        SimpleModel('/tmp/sm_save/', 'sm')

    def test_simple_model_prediction(self):
        """Test that SimpleModel adds 3 to input."""
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        dataset = DictionaryDataset({'x': np.array([[3], [4]])})

        output_dict = sm.predict(dataset)

        print(output_dict)

        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]]))

    def test_simple_model_train(self):
        """SimpleModel should not be able to train. Confirm this."""
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        d = DictionaryDataset({'x': np.array([[3], [4]])})

        with self.assertRaises(ValueError):
            sm.train(d, num_epochs=10)

    def test_less_simple_model_train(self):
        """Trains LessSimpleModel for 10000 epochs to converge w value to 3."""
        with tf.Graph().as_default():
            lsm = LessSimpleModel('/tmp/lsm_save/', 'lsm')

            dataset = DictionaryDataset({'x': np.array([[3]]), 'label': np.array([[6]])})

            output_dict = lsm.train(dataset, output_tensor_names=['w'], num_epochs=10000, verbose=False)
            epsilon = .01
            assert np.abs(3 - output_dict['w']) < epsilon
            print(output_dict)

            output_dict = lsm.predict(dataset, ['y'])

            print(output_dict['y'])

            assert np.abs(output_dict['y'] - 6) < epsilon

    def test_even_less_simple_model_train(self):
        """Train EvenLessSimpleModel for 10000 epochs to converge w value to 3."""
        with tf.Graph().as_default():
            lsm = EvenLessSimpleModel('/tmp/lsm_save/', 'lsm')

            dataset = DictionaryDataset({'x': np.array([[3]]), 'label': np.array([[6]])})

            output_dict = lsm.train(dataset, output_tensor_names=['w'], num_epochs=10000, verbose=False)
            epsilon = .1
            assert np.abs(3 - output_dict['w']) < epsilon
            assert output_dict['loss'] < epsilon
            assert len(lsm.train_ops) == 1
            print(output_dict)

            output_dict = lsm.predict(dataset, ['y', 'w'])

            assert np.abs(output_dict['y'] - 6) < epsilon


