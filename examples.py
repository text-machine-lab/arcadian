import tensorflow as tf

from arcadian.gm import GenericModel


class SimpleModel(GenericModel):
    """Subclass of generic model that does not train, only performs static operation (adds 3 to input)."""
    def build(self):
        self.trainable = False
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_output = tf_input + 3.0
        self.i['x'] = tf_input
        self.o['y'] = tf_output

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        print('Executing action_per_epoch')

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
        print('Executing action_per_batch')

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None, batch_size=32,
                           train_op_names=None, save_per_epoch=True, **kwargs):
        return False  # Do not save after each epoch


class LessSimpleModel(GenericModel):
    """Subclass of generic model that learns a scalar value w based on training examples.
    Defines only the loss function, no optimizer. Learns to add a scalar to input."""
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_w = tf.get_variable('w', (1, 1), initializer=tf.contrib.layers.xavier_initializer())
        tf_output = tf_input + tf_w
        self.i['x'] = tf_input
        self.o['y'] = tf_output
        self.o['w'] = tf_w

        tf_label = tf.placeholder(tf.float32, shape=(None, 1), name='label')
        tf_loss = tf.nn.l2_loss(tf_label - self.o['y'])
        train_op = tf.train.AdamOptimizer(.001).minimize(tf_loss)
        self.i['label'] = tf_label
        self.o['loss'] = tf_loss
        self.train_ops['l2_loss'] = train_op

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None,
                               batch_size=32, train_op_names=None, **kwargs):
        self.save_per_epoch = False


class EvenLessSimpleModel(GenericModel):
    """Subclass of generic model that learns a scalar value w based on training examples.
    User defines the Adam optimizer internally, making it even less simple. Learns to add
    a scalar to input."""
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_w = tf.get_variable('w', (1, 1), initializer=tf.contrib.layers.xavier_initializer())
        tf_output = tf_input + tf_w
        self.i['x'] = tf_input
        self.o['y'] = tf_output
        self.o['w'] = tf_w

        tf_label = tf.placeholder(tf.float32, shape=(None, 1), name='label')
        self.loss = tf.nn.l2_loss(tf_label - self.o['y'])
        self.i['label'] = tf_label

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None,
                               batch_size=32, train_op_names=None, **kwargs):
        self.save_per_epoch = False