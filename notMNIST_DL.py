from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / predictions.shape[0])


def logistci_reggression_using_GD():
    train_subset = 10000
    graph = tf.Graph()

    with graph.as_default():
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # general weight. (a normal distribution)
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        logits = tf.matmul(tf_train_dataset, weights) + biases

        # loss Fcuntion = cross entropy between logits and one-hot labels
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # learning rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # e.g. softmax(logits) = [[0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.080.28], ...]
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    # train a logistic regresstion classifier
    num_steps = 801
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            _, l, predictions = session.run([optimizer, loss, train_prediction])

            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


def logistic_reggression_using_SGD():
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_data_set = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        logits = tf.matmul(tf_train_data_set, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    num_step = 3001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_step):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_data_set: batch_data, tf_train_labels: batch_labels}
            _ ,l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def one_hidden_layer_nn():
    batch_size = 128
    hidden_layer_num = 1000

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_num]))
        biases = tf.Variable(tf.zeros(hidden_layer_num))
        hidden_weights = tf.Variable(tf.truncated_normal([hidden_layer_num, num_labels]))
        hidden_biases = tf.Variable(tf.zeros(num_labels))

        hidden_logits = tf.matmul(tf_train_dataset, weights) + biases
        relu = tf.nn.relu(hidden_logits)
        output_logits = tf.matmul(relu, hidden_weights) + hidden_biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=tf_train_labels))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(output_logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu((tf.matmul(tf_valid_dataset, weights) + biases)), hidden_weights) + hidden_biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu((tf.matmul(tf_test_dataset, weights) + biases)), hidden_weights) + hidden_biases)

        num_step = 3001

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_step):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# logistci_reggression_using_GD()
# logistic_reggression_using_SGD()
one_hidden_layer_nn()