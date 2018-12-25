import numpy as np
from copy import deepcopy
from sympy.utilities.iterables import multiset_permutations
from flask import Flask, render_template, request, flash, redirect, url_for
import csv
from form import Form
import random
import tensorflow as tf
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


def get_accuracy():
    return np.genfromtxt('accuracies.csv', delimiter=',', dtype="float32")


def neural_network_model_shallow(data, train_x):
    n_nodes_hl1 = 1000
    n_nodes_hl2 = 700
    n_nodes_hl3 = 500

    n_classes = 2

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']) # layer 1 output
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']) # layer 2 output
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']) # layer 3 output
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases'] # output

    return output


def neural_network_model_deep(data, train_x):
    n_nodes_hl1 = 1000
    n_nodes_hl2 = 1000
    n_nodes_hl3 = 700
    n_nodes_hl4 = 700
    n_nodes_hl5 = 500

    n_classes = 2

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']) # layer 1 output
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']) # layer 2 output
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']) # layer 3 output
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases']) # layer 4 output
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases']) # layer 5 output
    l5 = tf.nn.relu(l5)

    output = tf.matmul(l5, output_layer['weights']) + output_layer['biases'] # output

    return output


def train_complex_snn(train_x, train_y, test_x, test_y):
    x_complex = tf.placeholder('float', [None, len(train_x[0])])
    y_complex = tf.placeholder('float')

    batch_size = 500

    prediction = neural_network_model_shallow(x_complex, train_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_complex))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    how_many_epochs = 30

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        saver_complex = tf.train.Saver()
        epoch = 1

        while epoch <= how_many_epochs:
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                # if epoch != 1:
                #     saver_complex.restore(sess, "./model_complex.ckpt")
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x_complex: batch_x, y_complex: batch_y})
                epoch_loss += c
                i += batch_size


            saver_complex.save(sess, "./model_complex/model_complex.ckpt")
            print('Epoch', epoch, 'completed out of', how_many_epochs, 'loss:', epoch_loss)
            epoch += 1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_complex, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_complex_snn = accuracy.eval({x_complex: test_x, y_complex: test_y})
        print('Accuracy:', accuracy_complex_snn)
        return accuracy_complex_snn


def train_complex_dnn(train_x, train_y, test_x, test_y):
    x_complex = tf.placeholder('float', [None, len(train_x[0])])
    y_complex = tf.placeholder('float')

    batch_size = 500

    prediction = neural_network_model_deep(x_complex, train_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_complex))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    how_many_epochs = 50

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        saver_complex_dnn = tf.train.Saver()
        epoch = 1

        while epoch <= how_many_epochs:
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                # if epoch != 1:
                #     saver_complex.restore(sess, "./model_complex.ckpt")
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x_complex: batch_x, y_complex: batch_y})
                epoch_loss += c
                i += batch_size


            saver_complex_dnn.save(sess, "./model_complex_deep/model_complex_deep.ckpt")
            print('Epoch', epoch, 'completed out of', how_many_epochs, 'loss:', epoch_loss)
            epoch += 1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_complex, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_complex_dnn = accuracy.eval({x_complex: test_x, y_complex: test_y})
        print('Accuracy:', accuracy_complex_dnn)
        return accuracy_complex_dnn


def train_simple_dnn(train_x, train_y, test_x, test_y):
    x_simple = tf.placeholder('float', [None, len(train_x[0])])
    y_simple = tf.placeholder('float')

    batch_size = 500

    prediction = neural_network_model_deep(x_simple, train_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_simple))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    how_many_epochs = 35

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        epoch = 1
        saver_simple_dnn = tf.train.Saver()

        while epoch <= how_many_epochs:
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                # if epoch != 1:
                #     saver_complex.restore(sess, "./model_complex.ckpt")
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x_simple: batch_x, y_simple: batch_y})
                epoch_loss += c
                i += batch_size


            saver_simple_dnn.save(sess, "./model_simple_deep/model_simple_deep.ckpt")
            print('Epoch', epoch, 'completed out of', how_many_epochs, 'loss:', epoch_loss)
            epoch += 1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_simple, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_simple_dnn = accuracy.eval({x_simple: test_x, y_simple: test_y})
        print('Accuracy:', accuracy_simple_dnn)
        return accuracy_simple_dnn


def train_simple_snn(train_x, train_y, test_x, test_y):
    x_simple = tf.placeholder('float', [None, len(train_x[0])])
    y_simple = tf.placeholder('float')

    batch_size = 500

    prediction = neural_network_model_shallow(x_simple, train_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_simple))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    how_many_epochs = 30

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        epoch = 1
        saver_simple = tf.train.Saver()

        while epoch <= how_many_epochs:
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                # if epoch != 1:
                #     saver_complex.restore(sess, "./model_complex.ckpt")
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x_simple: batch_x, y_simple: batch_y})
                epoch_loss += c
                i += batch_size

            saver_simple.save(sess, "./model_simple/model_simple.ckpt")
            print('Epoch', epoch, 'completed out of', how_many_epochs, 'loss:', epoch_loss)
            epoch += 1

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_simple, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_simple_snn = accuracy.eval({x_simple: test_x, y_simple: test_y})
        print('Accuracy:', accuracy_simple_snn)
        return accuracy_simple_snn


def create_feature_sets_and_labels_complex_nn():
    game_data = np.genfromtxt('game_data_normalized.csv', delimiter=',', dtype="float32")

    testing_size = int(.1 * len(game_data))
    train_y = []
    for i in game_data[:-testing_size]:
        if int(i[2]) == 0:
            train_y.append([1, 0])
        elif int(i[2]) == 1:
            train_y.append([0, 1])

    test_y = []
    for i in game_data[-testing_size:]:
        if int(i[2]) == 0:
            test_y.append([1, 0])
        elif int(i[2]) == 1:
            test_y.append([0, 1])


    train_x = []
    for i in game_data[:-testing_size]:
        train_x.append(list(i[np.arange(len(i))!=2]))

    test_x = []
    for i in game_data[-testing_size:]:
        test_x.append(list(i[np.arange(len(i))!=2]))


    return train_x, train_y, test_x, test_y


def create_feature_sets_and_labels_complex_lr(file_name):
    print('Loading LR Complex')
    game_data = np.genfromtxt(file_name, delimiter=',', dtype="float32")

    testing_size = int(.5 * len(game_data))

    train_y = list(game_data[:, 2][:-testing_size])

    test_y = list(game_data[:, 2][:-testing_size])


    train_x = []
    for i in game_data[:-testing_size]:
        train_x.append(list(i[np.arange(len(i))!=2]))

    test_x = []
    for i in game_data[-testing_size:]:
        test_x.append(list(i[np.arange(len(i))!=2]))

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def train_logistic_regression_complex(x_train, y_train, x_test, y_test):
    print('Starting Complex Logistic Regression')
    log_reg = LogisticRegression(C=100)
    log_reg.fit(x_train, y_train)

    acc_train = log_reg.score(x_train, y_train)
    acc_test = log_reg.score(x_test, y_test)
    print('Accuracy:', (acc_train + acc_test) / 2)

    accuracy_complex_lr = (acc_train + acc_test) / 2
    dump(log_reg, 'log_reg_model_complex.joblib')
    return accuracy_complex_lr


def train_svm_complex(x_train, y_train, x_test, y_test):
    print('Starting SVM Complex Training')
    svmModel = svm.SVC(kernel='poly', gamma='scale', cache_size=7000)
    svmModel.fit(x_train, y_train)
    acc_train = svmModel.score(x_train, y_train)
    acc_test = svmModel.score(x_test, y_test)
    print('Accuracy:', (acc_train + acc_test) / 2)

    accuracy_complex_svm = (acc_train + acc_test) / 2

    dump(svmModel, 'svm_model_complex.joblib')
    return accuracy_complex_svm


def train_svm_simple(x_train, y_train, x_test, y_test):
    print('Starting SVM Simple Training')
    svmModel = svm.SVC(kernel='poly', gamma='scale', cache_size=7000)
    svmModel.fit(x_train, y_train)
    acc_train = svmModel.score(x_train, y_train)
    acc_test = svmModel.score(x_test, y_test)
    print('Accuracy:', (acc_train + acc_test) / 2)

    accuracy_simple_svm = (acc_train + acc_test) / 2

    dump(svmModel, 'svm_model_simple.joblib')
    return accuracy_simple_svm



def use_neural_network_complex_deep(input_data):
    x = tf.placeholder('float', [None, len(input_data)])
    y = tf.placeholder('float')

    print(input_data)

    prediction = neural_network_model_deep(x, [input_data])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        how_many_epochs = 30

        saver_complex_dnn = tf.train.Saver()
        saver_complex_dnn.restore(sess, "./model_complex_deep/model_complex_deep.ckpt")

        input_data = np.array(list(input_data))

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)))

        if result[0] == 0:
            print('Defeat:', input_data)
            return 'Defeat'
        elif result[0] == 1:
            print('Victory:', input_data)
            return 'Victory'


def use_neural_network_simple_deep(input_data):
    x = tf.placeholder('float', [None, len(input_data)])
    y = tf.placeholder('float')
    prediction = neural_network_model_deep(x, [input_data])

    print(input_data)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        how_many_epochs = 30

        saver_simple_dnn = tf.train.Saver()
        saver_simple_dnn.restore(sess, "./model_simple_deep/model_simple_deep.ckpt")

        input_data = np.array(list(input_data))

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)))

        if result[0] == 0:
            print('Defeat:', input_data)
            return 'Defeat'
        elif result[0] == 1:
            print('Victory:', input_data)
            return 'Victory'


def use_neural_network_simple_shallow(input_data):
    x = tf.placeholder('float', [None, len(input_data)])
    y = tf.placeholder('float')
    prediction = neural_network_model_shallow(x, [input_data])

    print(input_data)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        how_many_epochs = 30

        saver_simple = tf.train.Saver()
        saver_simple.restore(sess, "./model_simple/model_simple.ckpt")

        input_data = np.array(list(input_data))

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)))

        if result[0] == 0:
            print('Defeat:', input_data)
            return 'Defeat'
        elif result[0] == 1:
            print('Victory:', input_data)
            return 'Victory'


def use_neural_network_complex_shallow(input_data):
    x = tf.placeholder('float', [None, len(input_data)])
    y = tf.placeholder('float')
    prediction = neural_network_model_shallow(x, [input_data])

    print(input_data)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        how_many_epochs = 30

        saver_complex = tf.train.Saver()
        saver_complex.restore(sess, "./model_complex/model_complex.ckpt")

        input_data = np.array(list(input_data))

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)))

        if result[0] == 0:
            print('Defeat:', input_data)
            return 'Defeat'
        elif result[0] == 1:
            print('Victory:', input_data)
            return 'Victory'


def create_feature_sets_and_labels_simple_nn():
    game_data = np.genfromtxt('game_data_normalized.csv', delimiter=',', dtype="float32")

    testing_size = int(.1 * len(game_data))

    simplified_data = []

    for game in game_data:
        simplified_data.append(list(np.concatenate((game[0:3], game[15:]), axis=0)))

    simplified_data = np.array(simplified_data)
    train_y = []
    for i in simplified_data[:-testing_size]:
        if int(i[2]) == 0:
            train_y.append([1, 0])
        elif int(i[2]) == 1:
            train_y.append([0, 1])

    test_y = []
    for i in simplified_data[-testing_size:]:
        if int(i[2]) == 0:
            test_y.append([1, 0])
        elif int(i[2]) == 1:
            test_y.append([0, 1])

    train_x = []
    for i in simplified_data[:-testing_size]:
        train_x.append(list(i[np.arange(len(i))!=2]))

    test_x = []
    for i in simplified_data[-testing_size:]:
        test_x.append(list(i[np.arange(len(i))!=2]))


    return train_x, train_y, test_x, test_y


def create_feature_sets_and_labels_simple_lr(file_name):
    print('Loading Simple LR')
    game_data = np.genfromtxt(file_name, delimiter=',', dtype="float32")

    testing_size = int(.5 * len(game_data))

    simplified_data = []

    for game in game_data:
        simplified_data.append(list(np.concatenate((game[0:3], game[15:]), axis=0)))

    simplified_data = np.array(simplified_data)

    train_y = list(simplified_data[:, 2][:-testing_size])

    test_y = list(simplified_data[:, 2][:-testing_size])

    train_x = []
    for i in simplified_data[:-testing_size]:
        train_x.append(list(i[np.arange(len(i))!=2]))

    test_x = []
    for i in simplified_data[-testing_size:]:
        test_x.append(list(i[np.arange(len(i))!=2]))

    train_x = np.array(train_x)
    test_x = np.array(test_x)

    train_y = np.array(train_y)
    test_y = np.array(test_y)


    return train_x, train_y, test_x, test_y


def train_logistic_regression_simple(x_train, y_train, x_test, y_test):
    print('Starting Simple Logistic Regression')
    log_reg = LogisticRegression(C=100)
    log_reg.fit(x_train, y_train)

    acc_train = log_reg.score(x_train, y_train)
    acc_test = log_reg.score(x_test, y_test)
    print('Accuracy:', (acc_train + acc_test) / 2)

    accuracy_simple_lr = (acc_train + acc_test) / 2
    dump(log_reg, 'log_reg_model_simple.joblib')
    return accuracy_simple_lr


def numerize(game_data):
    height = game_data.shape[0]
    length = game_data.shape[1]

    for i in range(height):
        for j in range(length):
            if game_data[i][j] in b'd':
                game_data[i][j] = 0
            elif game_data[i][j] in b'v':
                game_data[i][j] = 1
            elif game_data[i][j] in b'a':
                game_data[i][j] = 1
            elif game_data[i][j] in b'anub':
                game_data[i][j] = 0
            elif game_data[i][j] in b'blizz':
                game_data[i][j] = 1
            elif game_data[i][j] in b'busan':
                game_data[i][j] = 2
            elif game_data[i][j] in b'dorado':
                game_data[i][j] = 3
            elif game_data[i][j] in b'eich':
                game_data[i][j] = 4
            elif game_data[i][j] in b'gibr':
                game_data[i][j] = 5
            elif game_data[i][j] in b'ana':
                game_data[i][j] = 24
            elif game_data[i][j] in b'hana':
                game_data[i][j] = 6
            elif game_data[i][j] in b'holly':
                game_data[i][j] = 7
            elif game_data[i][j] in b'hor':
                game_data[i][j] = 8
            elif game_data[i][j] in b'ill':
                game_data[i][j] = 9
            elif game_data[i][j] in b'junk':
                game_data[i][j] = 10
            elif game_data[i][j] in b'king':
                game_data[i][j] = 11
            elif game_data[i][j] in b'lijiang':
                game_data[i][j] = 12
            elif game_data[i][j] in b'nepal':
                game_data[i][j] = 13
            elif game_data[i][j] in b'numb':
                game_data[i][j] = 14
            elif game_data[i][j] in b'oasis':
                game_data[i][j] = 15
            elif game_data[i][j] in b'rial':
                game_data[i][j] = 16
            elif game_data[i][j] in b'route':
                game_data[i][j] = 17
            elif game_data[i][j] in b'vols':
                game_data[i][j] = 18
            elif game_data[i][j] in b'none':
                game_data[i][j] = 0
            elif game_data[i][j] in b'dva':
                game_data[i][j] = 1
            elif game_data[i][j] in b'orisa':
                game_data[i][j] = 2
            elif game_data[i][j] in b'rein':
                game_data[i][j] = 3
            elif game_data[i][j] in b'roadhog':
                game_data[i][j] = 4
            elif game_data[i][j] in b'winston':
                game_data[i][j] = 5
            elif game_data[i][j] in b'hamond':
                game_data[i][j] = 6
            elif game_data[i][j] in b'zarya':
                game_data[i][j] = 7
            elif game_data[i][j] in b'ashe':
                game_data[i][j] = 8
            elif game_data[i][j] in b'bastion':
                game_data[i][j] = 9
            elif game_data[i][j] in b'doom':
                game_data[i][j] = 10
            elif game_data[i][j] in b'genji':
                game_data[i][j] = 11
            elif game_data[i][j] in b'hanzo':
                game_data[i][j] = 12
            elif game_data[i][j] in b'junkrat':
                game_data[i][j] = 13
            elif game_data[i][j] in b'mccree':
                game_data[i][j] = 14
            elif game_data[i][j] in b'mei':
                game_data[i][j] = 15
            elif game_data[i][j] in b'pharah':
                game_data[i][j] = 16
            elif game_data[i][j] in b'reaper':
                game_data[i][j] = 17
            elif game_data[i][j] in b'soldier':
                game_data[i][j] = 18
            elif game_data[i][j] in b'sombra':
                game_data[i][j] = 19
            elif game_data[i][j] in b'sym':
                game_data[i][j] = 20
            elif game_data[i][j] in b'torb':
                game_data[i][j] = 21
            elif game_data[i][j] in b'tracer':
                game_data[i][j] = 22
            elif game_data[i][j] in b'widow':
                game_data[i][j] = 23
            elif game_data[i][j] in b'brig':
                game_data[i][j] = 25
            elif game_data[i][j] in b'lucio':
                game_data[i][j] = 26
            elif game_data[i][j] in b'mercy':
                game_data[i][j] = 27
            elif game_data[i][j] in b'moira':
                game_data[i][j] = 28
            elif game_data[i][j] in b'zen':
                game_data[i][j] = 29


def switchTeams(game_data, game_data_team_switch):
    height = game_data.shape[0]

    for i in range(height):
        if game_data[i][0] in b'0':
            game_data[i][0] = 1
        elif game_data[i][0] in b'1':
            game_data[i][0] = 0

        if (game_data[i][1] not in b'2') and (game_data[i][1] not in b'9') and (game_data[i][1] not in b'13') and (game_data[i][1] not in b'12'):
            if game_data[i][2] in b'0':
                game_data[i][2] = 1
            elif game_data[i][2] in b'1':
                game_data[i][2] = 0

        team1 = deepcopy(game_data[i][3:9])
        team2 = deepcopy(game_data[i][9:15])
        team1_comp = deepcopy(game_data[i][15:18])
        team2_comp = deepcopy(game_data[i][18:21])

        game_data[i][3:9] = team2
        game_data[i][9:15] = team1
        game_data[i][15:18] = team2_comp
        game_data[i][18:21] = team1_comp
    game_data_team_switch = np.concatenate((game_data, game_data_team_switch), axis=0)
    return game_data_team_switch


def switchTeamsNewData(game_data, game_data_team_switch):
    game_data = np.array(game_data)
    game_data_team_switch = np.array(game_data_team_switch)

    height = game_data.shape[0]

    for i in range(height):
        if game_data[i][0] is 0:
            game_data[i][0] = 1
        elif game_data[i][0] is 1:
            game_data[i][0] = 0

        if (game_data[i][1] is not 2) and (game_data[i][1] is not 9) and (game_data[i][1] is not 13) and (game_data[i][1] is not 12):
            if game_data[i][2] is 0:
                game_data[i][2] = 1
            elif game_data[i][2] is 1:
                game_data[i][2] = 0

        team1 = deepcopy(game_data[i][3:9])
        team2 = deepcopy(game_data[i][9:15])
        team1_comp = deepcopy(game_data[i][15:18])
        team2_comp = deepcopy(game_data[i][18:21])

        game_data[i][3:9] = team2
        game_data[i][9:15] = team1
        game_data[i][15:18] = team2_comp
        game_data[i][18:21] = team1_comp
    game_data_team_switch = np.concatenate((game_data, game_data_team_switch), axis=0)
    return game_data_team_switch


def permutateTeams(game_data, new_data):
    height = game_data.shape[0]

    for i in range(height):
        team1 = deepcopy(game_data[i][3:9])
        team2 = deepcopy(game_data[i][9:15])
        beginning = deepcopy(game_data[i][0:3])
        end = deepcopy(game_data[i][15:])

        for team1_perm in list(multiset_permutations(team1))[0:65]:
            for team2_perm in list(multiset_permutations(team2))[0:65]:
                new_point = np.concatenate((beginning, team1_perm, team2_perm, end), axis=0)
                new_data = np.append(new_data, [new_point], axis=0)

    return new_data


def normalizeData(game_data):

    for data in game_data:
        data[1] = int(data[1]) / 18
        for i in range(3, 15):
            data[i] = int(data[i]) / 29
        for i in range(15, 21):
            data[i] = int(data[i]) / 6

    return game_data



def preprocess():
    game_data = np.genfromtxt('game_data.csv', delimiter=',', dtype="|S5")

    if 'd' in str(game_data[0][0]):
        game_data[0][0] = 'd'
    else:
        game_data[0][0] = 'v'

    numerize(game_data)
    game_data_team_switch = deepcopy(game_data)
    game_data_team_switch = switchTeams(game_data, game_data_team_switch)
    random.shuffle(game_data_team_switch)
    game_data_team_permutation = deepcopy(game_data_team_switch)
    game_data_team_permutation = permutateTeams(game_data_team_permutation, game_data_team_switch)
    game_data_normalized = normalizeData(game_data_team_permutation)
    random.shuffle(game_data_normalized)
    np.savetxt("game_data_normalized.csv", game_data_normalized.astype(str), delimiter=',', fmt='%s')

app = Flask(__name__)

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data_form = Form()
    outcome = ''

    data_form.position.choices = [('0', 'Attack'), ('1', 'Defense')]

    data_form.team_member2.choices = [(character[0], character[1]) for character in data_form.team_member1.choices]
    data_form.team_member3.choices = [(character[0], character[1]) for character in data_form.team_member1.choices]
    data_form.team_member4.choices = [(character[0], character[1]) for character in data_form.team_member1.choices]
    data_form.team_member5.choices = [(character[0], character[1]) for character in data_form.team_member1.choices]
    data_form.team_member6.choices = [(character[0], character[1]) for character in data_form.team_member1.choices]

    data_form.enemy_member2.choices = [(character[0], character[1]) for character in data_form.enemy_member1.choices]
    data_form.enemy_member3.choices = [(character[0], character[1]) for character in data_form.enemy_member1.choices]
    data_form.enemy_member4.choices = [(character[0], character[1]) for character in data_form.enemy_member1.choices]
    data_form.enemy_member5.choices = [(character[0], character[1]) for character in data_form.enemy_member1.choices]
    data_form.enemy_member6.choices = [(character[0], character[1]) for character in data_form.enemy_member1.choices]

    if request.method == 'POST' and validate(data_form) is 'Good':
        new_data = [[data_form.map.data,
                     data_form.position.data,
                     data_form.team_member1.data,
                     data_form.team_member2.data,
                     data_form.team_member3.data,
                     data_form.team_member4.data,
                     data_form.team_member5.data,
                     data_form.team_member6.data,
                     data_form.enemy_member1.data,
                     data_form.enemy_member2.data,
                     data_form.enemy_member3.data,
                     data_form.enemy_member4.data,
                     data_form.enemy_member5.data,
                     data_form.enemy_member6.data
                    ]]

        team_dps = 0
        enemy_dps = 0
        team_tanks = 0
        enemy_tanks = 0
        team_healers = 0
        enemy_healers = 0

        for i in range(len(new_data[0]) - 2):
            if i < 6:
                if int(new_data[0][i + 2]) > 0 and int(new_data[0][i + 2]) <= 7:
                    team_tanks += 1
                elif int(new_data[0][i + 2]) > 7 and int(new_data[0][i + 2]) <= 23:
                    team_dps += 1
                elif int(new_data[0][i + 2]) > 23 and int(new_data[0][i + 2]) != 0:
                    team_healers += 1
            else:
                if int(new_data[0][i + 2]) > 0 and int(new_data[0][i + 2]) <= 7:
                    enemy_tanks += 1
                elif int(new_data[0][i + 2]) > 7 and int(new_data[0][i + 2]) <= 23:
                    enemy_dps += 1
                elif int(new_data[0][i + 2]) > 23 and int(new_data[0][i + 2]) != 0:
                    enemy_healers += 1


        new_data[0].append(team_tanks)
        new_data[0].append(team_dps)
        new_data[0].append(team_healers)
        new_data[0].append(enemy_tanks)
        new_data[0].append(enemy_dps)
        new_data[0].append(enemy_healers)

        new_data[0][0] = int(new_data[0][0]) / 18
        new_data[0][1] = 0
        for i in range(2, 14):
            new_data[0][i] = int(new_data[0][i]) / 29
        for i in range(14, 20):
            new_data[0][i] = int(new_data[0][i]) / 6

        new_data = new_data[0]

        if data_form.algorithm.data is '0':
            outcome = use_neural_network_complex_deep(new_data)
        elif data_form.algorithm.data is '1':
            new_data = list(np.concatenate((new_data[0:2], new_data[14:20]), axis=0))
            outcome = use_neural_network_simple_deep(new_data)
        elif data_form.algorithm.data is '2':
            outcome = use_neural_network_complex_shallow(new_data)
        elif data_form.algorithm.data is '3':
            new_data = list(np.concatenate((new_data[0:2], new_data[14:20]), axis=0))
            outcome = use_neural_network_simple_shallow(new_data)
        elif data_form.algorithm.data is '4':
            new_data = np.array(new_data)
            svm_model = load('svm_model_complex.joblib')
            outcome = svm_model.predict([new_data])
            if outcome == [1.]:
                outcome = 'Victory'
            elif outcome == [0.]:
                outcome = 'Defeat'
        elif data_form.algorithm.data is '5':
            new_data = np.concatenate((new_data[0:2], new_data[14:20]), axis=0)
            svm_model = load('svm_model_simple.joblib')
            outcome = svm_model.predict([new_data])
            if outcome == [1.]:
                outcome = 'Victory'
            elif outcome == [0.]:
                outcome = 'Defeat'
        elif data_form.algorithm.data is '6':
            new_data = np.array(new_data)
            log_reg = load('log_reg_model_complex.joblib')
            outcome = log_reg.predict([new_data])
            if outcome == [1.]:
                outcome = 'Victory'
            elif outcome == [0.]:
                outcome = 'Defeat'
        elif data_form.algorithm.data is '7':
            new_data = np.concatenate((new_data[0:2], new_data[14:20]), axis=0)
            log_reg = load('log_reg_model_simple.joblib')
            outcome = log_reg.predict([new_data])
            if outcome == [1.]:
                outcome = 'Victory'
            elif outcome == [0.]:
                outcome = 'Defeat'

        if outcome is 'Victory':
            flash('The predicted outcome from this model is a Victory!', 'success')
        elif outcome is 'Defeat':
            flash('The predicted outcome from this model is a Defeat.', 'danger')

    else:
        if validate(data_form) is 'Position':
            flash('This position is not valid for the selected map', 'danger')
        elif validate(data_form) is 'Team':
            flash('You can not have multiples of the same character on your team', 'danger')
        elif validate(data_form) is 'Enemy':
            flash('You can not have multiples of the same character on the enemy team', 'danger')
    return render_template('predict.html', prediction=outcome, form=data_form)


def validate(form):
    try:
        form_team = [form.team_member1.data,
                     form.team_member2.data,
                     form.team_member3.data,
                     form.team_member4.data,
                     form.team_member5.data,
                     form.team_member6.data,
                     ]


        form_enemy = [form.enemy_member1.data,
                     form.enemy_member2.data,
                     form.enemy_member3.data,
                     form.enemy_member4.data,
                     form.enemy_member5.data,
                     form.enemy_member6.data,
                     ]

        if int(form.map.data) is 2 or int(form.map.data) is 9 or int(form.map.data) is 12 or int(form.map.data) is 13 or int(form.map.data) is 15:
            if int(form.position.data) is 1:
                return 'Position'

        for i in range(len(form_team) - 1):
            for j in range(i + 1, len(form_team)):
                if int(form_team[i]) is int(form_team[j]) and (form_team[i] is not '0'):
                    return 'Team'

        for i in range(len(form_enemy) - 1):
            for j in range(i + 1, len(form_enemy)):
                if int(form_enemy[i]) is int(form_enemy[j]) and (form_enemy[i] is not '0'):
                    return 'Enemy'
    except:
        return 'Incomplete'
    return 'Good'


@app.route('/new', methods=['GET', 'POST'])
def new():
    new_data_form = Form()

    new_data_form.position.choices = [('1', 'Attack'), ('0', 'Defense')]

    new_data_form.team_member2.choices = [(character[0], character[1]) for character in new_data_form.team_member1.choices]
    new_data_form.team_member3.choices = [(character[0], character[1]) for character in new_data_form.team_member1.choices]
    new_data_form.team_member4.choices = [(character[0], character[1]) for character in new_data_form.team_member1.choices]
    new_data_form.team_member5.choices = [(character[0], character[1]) for character in new_data_form.team_member1.choices]
    new_data_form.team_member6.choices = [(character[0], character[1]) for character in new_data_form.team_member1.choices]

    new_data_form.enemy_member2.choices = [(character[0], character[1]) for character in new_data_form.enemy_member1.choices]
    new_data_form.enemy_member3.choices = [(character[0], character[1]) for character in new_data_form.enemy_member1.choices]
    new_data_form.enemy_member4.choices = [(character[0], character[1]) for character in new_data_form.enemy_member1.choices]
    new_data_form.enemy_member5.choices = [(character[0], character[1]) for character in new_data_form.enemy_member1.choices]
    new_data_form.enemy_member6.choices = [(character[0], character[1]) for character in new_data_form.enemy_member1.choices]

    if request.method == 'POST' and validate(new_data_form) is 'Good':
        new_data = [[new_data_form.outcome.data,
                    new_data_form.map.data,
                    new_data_form.position.data,
                    new_data_form.team_member1.data,
                    new_data_form.team_member2.data,
                    new_data_form.team_member3.data,
                    new_data_form.team_member4.data,
                    new_data_form.team_member5.data,
                    new_data_form.team_member6.data,
                    new_data_form.enemy_member1.data,
                    new_data_form.enemy_member2.data,
                    new_data_form.enemy_member3.data,
                    new_data_form.enemy_member4.data,
                    new_data_form.enemy_member5.data,
                    new_data_form.enemy_member6.data
                    ]]

        team_dps = 0
        enemy_dps = 0
        team_tanks = 0
        enemy_tanks = 0
        team_healers = 0
        enemy_healers = 0

        for i in range(len(new_data[0]) - 3):
            if i < 6:
                if int(new_data[0][i + 3]) > 0 and int(new_data[0][i + 3]) <= 6:
                    team_healers += 1
                elif int(new_data[0][i + 3]) > 6 and int(new_data[0][i + 3]) <= 22:
                    team_dps += 1
                elif int(new_data[0][i + 3]) > 22 and int(new_data[0][i + 3]) != 0:
                    team_tanks += 1
            else:
                if int(new_data[0][i + 3]) > 0 and int(new_data[0][i + 3]) <= 6:
                    enemy_healers += 1
                elif int(new_data[0][i + 3]) > 6 and int(new_data[0][i + 3]) <= 22:
                    enemy_dps += 1
                elif int(new_data[0][i + 3]) > 22 and int(new_data[0][i + 3]) != 0:
                    enemy_tanks += 1

        new_data[0].append(team_tanks)
        new_data[0].append(team_dps)
        new_data[0].append(team_healers)
        new_data[0].append(enemy_tanks)
        new_data[0].append(enemy_dps)
        new_data[0].append(enemy_healers)

        new_data = normalizeData(new_data)
        new_data_team_switch = deepcopy(new_data)
        new_data_team_switch = switchTeamsNewData(new_data, new_data_team_switch)
        new_data_perm = deepcopy(new_data_team_switch)
        new_data_perm = permutateTeams(new_data_perm, new_data_team_switch)

        with open('game_data_normalized.csv', 'a') as newFile:
             newFileWriter = csv.writer(newFile)
             for data in new_data_perm:
                 newFileWriter.writerow(data)

        flash('New data successfuly added.', 'success')

    else:
        if validate(new_data_form) is 'Position':
            flash('This position is not valid for the selected map', 'danger')
        elif validate(new_data_form) is 'Team':
            flash('You can not have multiples of the same character on your team', 'danger')
        elif validate(new_data_form) is 'Enemy':
            flash('You can not have multiples of the same character on the enemy team', 'danger')

    return render_template('new.html', form=new_data_form)


@app.route('/train', methods=['GET'])
def train():
    train_x_complex_dnn, train_y_complex_dnn, test_x_complex_dnn, test_y_complex_dnn = create_feature_sets_and_labels_complex_nn()
    accuracy_complex_snn = train_complex_snn(train_x_complex_dnn, train_y_complex_dnn, test_x_complex_dnn, test_y_complex_dnn)
    print(accuracy_complex_snn)
    tf.reset_default_graph()
    train_x_simple_dnn, train_y_simple_dnn, test_x_simple_dnn, test_y_simple_dnn = create_feature_sets_and_labels_simple_nn()
    accuracy_simple_snn = train_simple_snn(train_x_simple_dnn, train_y_simple_dnn, test_x_simple_dnn, test_y_simple_dnn)
    print(accuracy_simple_snn)
    tf.reset_default_graph()
    accuracy_complex_dnn = train_complex_dnn(train_x_complex_dnn, train_y_complex_dnn, test_x_complex_dnn, test_y_complex_dnn)
    print(accuracy_complex_dnn)
    tf.reset_default_graph()
    accuracy_simple_dnn = train_simple_dnn(train_x_simple_dnn, train_y_simple_dnn, test_x_simple_dnn, test_y_simple_dnn)
    print(accuracy_simple_dnn)
    tf.reset_default_graph()
    train_x_lr_complex, train_y_lr_complex, test_x_lr_complex, test_y_lr_complex = create_feature_sets_and_labels_complex_lr('game_data_normalized.csv')
    accuracy_complex_lr = train_logistic_regression_complex(train_x_lr_complex, train_y_lr_complex, test_x_lr_complex, test_y_lr_complex)
    print(accuracy_complex_lr)
    train_x_lr_simple, train_y_lr_simple, test_x_lr_simple, test_y_lr_simple = create_feature_sets_and_labels_simple_lr('game_data_normalized.csv')
    accuracy_simple_lr = train_logistic_regression_simple(train_x_lr_simple, train_y_lr_simple, test_x_lr_simple, test_y_lr_simple)
    print(accuracy_simple_lr)
    train_x_svm_complex, train_y_svm_complex, test_x_svm_complex, test_y_svm_complex = create_feature_sets_and_labels_complex_lr('game_data_normalized_small.csv')
    accuracy_complex_svm = train_svm_complex(train_x_svm_complex, train_y_svm_complex, test_x_svm_complex, test_y_svm_complex)
    print(accuracy_complex_svm)
    train_x_svm_simple, train_y_svm_simple, test_x_svm_simple, test_y_svm_simple = create_feature_sets_and_labels_simple_lr('game_data_normalized_small.csv')
    accuracy_simple_svm = train_svm_simple(train_x_svm_simple, train_y_svm_simple, test_x_svm_simple, test_y_svm_simple)
    print(accuracy_simple_svm)
    accuracies = np.array([accuracy_complex_snn, accuracy_simple_snn, accuracy_complex_dnn, accuracy_simple_dnn, accuracy_complex_lr, accuracy_simple_lr, accuracy_complex_svm, accuracy_simple_svm])
    print(accuracies)
    np.savetxt("accuracies.csv", accuracies, delimiter=',', fmt='%s')
    return render_template('train.html')


@app.route('/comparison')
def compare():
    accuracies = get_accuracy()

    objects = ['5LDNN-C', '5LDNN-S', '7LDNN-C', '7LDNN-S', 'LR-C', 'LR-S', 'SVM-C', 'SVM-S']
    y_pos = np.arange(len(objects))
    accuracies = list(accuracies)

    plt.bar(y_pos, accuracies, align='center', alpha=.5)
    plt.xticks(y_pos, objects, rotation=30)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    axes = plt.gca()
    axes.set_ylim([.5, 1.01])
    plt.savefig('./static/accuracies.png')

    return render_template('comparison.html',
                           accuracy1=accuracies[0],
                           accuracy2=accuracies[1],
                           accuracy3=accuracies[2],
                           accuracy4=accuracies[3],
                           accuracy5=accuracies[4],
                           accuracy6=accuracies[5],
                           accuracy7=accuracies[6],
                           accuracy8=accuracies[7]
                           )


if __name__ == '__main__':
    # preprocess()
    app.run()
