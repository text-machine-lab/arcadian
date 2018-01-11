"""David Donahue November 2017. This is a generic parent class for Tensorflow models. This file should be stand-alone."""
import time
from abc import ABC, abstractmethod
import os
import tensorflow as tf
import arcadian.dataset
from tqdm import tqdm
import numpy as np


class GenericModel(ABC):
    def __init__(self, save_dir=None, tensorboard_name=None, restore_from_save=False, trainable=True, tf_log_level='2'):
        """Abstract class which contains support functionality for developing Tensorflow models.
        Derive subclasses from this class and override the build() method. Define entire model in this method,
        and add all placeholders to self.input_placeholders dictionary as name:placeholder pairs. Add all tensors
        you wish to evaluate to self.output_tensors as name:tensor pairs. Assign loss function you wish to train on
        to self.loss variable ('loss' tensor automatically added to self.output_tensors). Names chosen for placeholders
        and output tensors are used to refer to these tensors outside of the model. As a general rule, Tensorflow
        should not need to be imported by the user of this object. Once model is defined, training and prediction are
        already implemented. Model graph is automatically saved to Tensorboard directory. Model parameters are
        automatically saved after each epoch, and restored from save after training if desired. If restored,
        entire model can be loaded, or only specific scopes by adding scopes to self.load_scopes list. Variable
        initialization and session maintenance are handled internally. This model supports custom operations
        if necessary.

        Arguments:
            save_dir: directory with which to save your model checkpoint files to
            tensorboard_name: directory name in /tmp/ where Tensorboard graph will be saved
            restore_from_save: indicates whether to restore model from save
            trainable: decide whether model will be trainable
            tf_log_level: by default, disables all outputs from Tensorflow backend (except errors)
        """
        assert not restore_from_save or trainable  # don't restore un-trainable model
        assert not restore_from_save or save_dir is not None  # can only restore if there is a save directory

        self.save_per_epoch = (save_dir is not None and trainable)
        self.shuffle = True
        self.restore_from_save = restore_from_save

        # Make sure there is slash after save directory. Important!
        if save_dir is not None and not save_dir.endswith('/'):
            save_dir += '/'

        self.save_dir = save_dir

        # Create directory to save model
        if save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Use Just-In-Time Compilation
        self.config = tf.ConfigProto()
        self.config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.trainable = trainable
            self.tensorboard_name = tensorboard_name
            self.load_scopes = []
            self.inputs = {}
            self.outputs = {}
            self.train_ops = []
            self.loss = None
            self._create_standard_placeholders()

            self.build()
            self._initialize_loss()
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level
            if self.tensorboard_name is not None:
                create_tensorboard_visualization(self.tensorboard_name)
            self.init = tf.global_variables_initializer()
            self.sess = tf.InteractiveSession(config=self.config)
            self.sess.run(self.init)
            if self.trainable:
                self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
            else:
                self.saver = None
            if self.save_dir is not None and self.restore_from_save:
                for each_scope in self.load_scopes:
                    load_scope_from_save(self.save_dir, self.sess, each_scope)
                if len(self.load_scopes) == 0:
                    restore_model_from_save(self.save_dir, self.sess)

    def _create_standard_placeholders(self):
        """"""
        self.inputs['is_training'] = tf.placeholder_with_default(False, (), name='is_training')

    def _fill_standard_placeholders(self, is_training):
        # Must at least return empty dictionary.
        return {'is_training': is_training}

    def _initialize_loss(self):
        """If user specifies loss to train on (using self.loss), create an Adam optimizer to minimize that loss,
        and add the optimizer operation to dictionary of train ops. Initialize optional placeholder
        for learning rate and add it to input placeholders under name 'learning rate'. Add loss tensor
        to output tensor dictionary under name 'loss', so it can be evaluated during training."""
        if self.loss is not None:
            learning_rate = tf.placeholder_with_default(.001, shape=(), name='learning_rate')
            self.train_ops.append(tf.train.AdamOptimizer(learning_rate).minimize(self.loss))
            self.inputs['learning rate'] = learning_rate
            self.outputs['loss'] = self.loss

    @abstractmethod
    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""
        pass

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, parameter_dict, **kwargs):
        """Optional: Define action to take place at the end of every epoch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this. Returns true to continue training. Only return false if you wish to
        implement early-stopping."""
        return True

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         parameter_dict, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""
        pass

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names,
                               parameter_dict, batch_size=32, **kwargs):
        """Optional: Define action to take place at the beginning of training/prediction, once. This could be
        used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
        pass

    def _eval(self, dataset, num_epochs, parameter_dict=None, output_tensor_names=None, batch_size=32, is_training=True, verbose=True, **kwargs):
        """Evaluate output tensors of model with dataset as input. Optionally train on that dataset. Return dictionary
        of evaluated tensors to user. For internal use only, shared functionality between training and prediction."""

        if verbose and is_training:
            print('Training...')

        if dataset is None:
            dataset = arcadian.dataset.EmptyDataset()

        # Allow user to give dictionaries of numpy features as input!
        if isinstance(dataset, dict):
            dataset = arcadian.dataset.DictionaryDataset(dataset)

        # Join parameters and default parameters into one dictionary
        united_parameter_dict = self._fill_standard_placeholders(is_training)
        if parameter_dict is not None:
            united_parameter_dict.update(parameter_dict)

        # Only train if model is trainable
        if is_training and not self.trainable:
            raise ValueError('Cannot train while model is not trainable.')

        # If user doesn't specify output tensors, evaluate them all!
        # Note: train() function doesn't allow output_tensor_names=None for simplicity
        if output_tensor_names is None:
            output_tensor_names = [name for name in self.outputs]

        self.action_before_training(dataset, num_epochs, is_training, output_tensor_names, parameter_dict,
                                    batch_size=batch_size, **kwargs)

        # Control what train ops are executed via arguments
        if is_training:
            train_op_list = self.train_ops
        else:
            train_op_list = []

        # Loss should always be evaluated during training if it exists
        if self.loss is not None and is_training:
            if 'loss' not in output_tensor_names:
                output_tensor_names.append('loss')

        # Create list of output tensors, initialize output dictionaries
        output_tensors = [self.outputs[each_tensor_name] for each_tensor_name in output_tensor_names]
        all_output_batch_dicts = None

        # Create feed dictionary for model parameters
        parameter_feed_dict = {self.inputs[feature_name]: united_parameter_dict[feature_name]
                               for feature_name in united_parameter_dict}

        continue_training = True
        do_shuffle = self.shuffle and is_training

        def optional_tqdm(iterable, verbose=True):
            """Function to disable tqdm output if verbose is disabled."""
            if verbose:
                for element in tqdm(iterable):
                    yield element
            else:
                for element in iterable:
                    yield element


        with self.graph.as_default():
            # Evaluate and/or train on dataset. Run user-defined action functions
            for epoch_index in range(num_epochs):
                if not continue_training:
                    break

                epoch_start_time = time.time()

                all_output_batch_dicts = []
                for batch_index, batch_dict in optional_tqdm(enumerate(dataset.generate_batches(batch_size=batch_size, shuffle=do_shuffle)),
                                                             verbose=(verbose and is_training)):
                    # Run batch in session - combine dataset features and parameters
                    feed_dict = {self.inputs[feature_name]: batch_dict[feature_name]
                                 for feature_name in batch_dict}
                    feed_dict.update(parameter_feed_dict)

                    if len(train_op_list) == 0:
                        train_op_list.append([]) # empty optimizer

                    # Execute all optimizers in order
                    output_numpy_arrays = None
                    for each_op in train_op_list:
                        output_numpy_arrays, _ = self.sess.run([output_tensors, each_op], feed_dict)

                    input_batch_dict = {feature_name: feed_dict[self.inputs[feature_name]]
                                        for feature_name in batch_dict}
                    output_batch_dict = {output_tensor_names[index]: output_numpy_arrays[index]
                                         for index in range(len(output_tensor_names))}

                    # Save evaluated tensors only for last optimizer run

                    # Keep history of batch outputs
                    all_output_batch_dicts.append(output_batch_dict)

                    self.action_per_batch(input_batch_dict, output_batch_dict, epoch_index,
                                          batch_index, is_training, parameter_dict, **kwargs)

                if self.save_per_epoch and self.trainable and is_training:
                    self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

                epoch_end_time = time.time()

                if is_training and verbose:
                    print('Epoch %s Elapsed Time: %s' % (epoch_index, epoch_end_time - epoch_start_time))

                # Calculate output dictionary from last epoch executed
                output_dict_concat = arcadian.dataset.concatenate_batch_dictionaries(all_output_batch_dicts)

                # Call user action per epoch, and allow them to stop training early
                continue_training = self.action_per_epoch(output_dict_concat, epoch_index, is_training,
                                                          parameter_dict, **kwargs)
                if not continue_training:
                    break

        return output_dict_concat

    def train(self, dataset, output_tensor_names=None, num_epochs=5, **kwargs):
        """Train on a dataset. Can specify which output tensors to evaluate (or none at all if dataset is too large).
        Can specify batch size and provide parameters arguments as inputs to model placeholders. To add constant
        values for input placeholders, pass to parameter_dict a dictionary containing name:value pairs. Name must
        match internal name of desired placeholder as defined in self.input_placeholders dictionary. Can set number
        of epochs to train for. **kwargs can be used to provide additional arguments to internal action functions,
        which can be overloaded for extra functionality. Training examples are shuffled each epoch!

        Arguments:
            dataset - subclass object of Dataset class containing labelled input features. Can also be dictionary
            output_tensor_names - list of names of output tensors to evaluate. Names defined in build() function
            num_epochs - number of epochs to train on. Is possible to implement early stopping using action functions
            parameter_dict - dictionary of constant parameters to provide to model (like learning rates)
            batch_size - number of examples to train on at once
            kwargs - optional parameters sent to action functions for expanded functionality

        Returns: dictionary of evaluated output tensors.
        """
        # For training: if user doesn't specify output tensors to evaluate, don't evaluate any.
        # If user wishes to evaluate all tensors, try output_tensor_names=(model).outputs
        if output_tensor_names is None:
            output_tensor_names = []

        output_tensor_dict = self._eval(dataset, num_epochs,
                                        output_tensor_names=output_tensor_names,
                                        is_training=True,
                                        **kwargs)

        return output_tensor_dict

    def predict(self, dataset, output_tensor_names=None, **kwargs):
        """Predict on a dataset. Can specify which output tensors to evaluate. Can specify batch size and provide
        parameters arguments as inputs to model placeholders. To add constant values for input placeholders, pass to
        parameter_dict a dictionary containing name:value pairs. Name must match internal name of desired placeholder
        as defined in self.input_placeholders dictionary. **kwargs can be used to provide additional arguments to
        internal action functions, which can be overloaded for extra functionality.

        Arguments:
            dataset - subclass object of Dataset class containing labelled input features. Can also be dictionary
            output_tensor_names - list of names of output tensors to evaluate. Names defined in build() function
            parameter_dict - dictionary of constant parameters to provide to model (like learning rates)
            batch_size - number of examples to train on at once
            kwargs - optional parameters sent to action functions for expanded functionality

        Returns: dictionary of evaluated output tensors."""
        output_tensor_dict = self._eval(dataset, 1,
                                        output_tensor_names=output_tensor_names,
                                        train_op_names=[],
                                        is_training=False,
                                        **kwargs)

        return output_tensor_dict


def create_tensorboard_visualization(model_name):
    """Saves the Tensorflow graph of your model, so you can view it in a TensorBoard console."""
    writer = tf.summary.FileWriter("/tmp/" + model_name + "/")
    writer.add_graph(tf.get_default_graph())
    return writer


def restore_model_from_save(model_var_dir, sess, var_list=None):
    """Restores all model variables from the specified directory."""
    if var_list is None:
        var_list = tf.trainable_variables()
    saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(model_var_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1


def load_scope_from_save(save_dir, sess, scope):
    """Load the encoder model variables from checkpoint in save_dir.
    Store them in session sess."""
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(variables) > 0
    restore_model_from_save(save_dir, sess, var_list=variables)



