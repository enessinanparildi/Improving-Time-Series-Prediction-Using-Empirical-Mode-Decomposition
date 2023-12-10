# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 23:57:44 2017

@author: Ben
"""

from PyEMD import EMD
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


def read_data(datadirectory, save_dir):
    dataset1 = "ibmstockdata.txt"
    dataset2 = "indianstockexchange.txt"
    dataset3 = "applestockbiggest.txt"

    ibmdatasrt = np.loadtxt(datadirectory + dataset1)
    s = ibmdatasrt
    plt.plot(s)
    plt.grid(True)
    plt.title(" IBM Dataset")
    plt.xlabel("Time")
    plt.ylabel("Target")

    indiandata = np.loadtxt(datadirectory + dataset2)
    s = indiandata
    plt.plot(indiandata)
    plt.grid(True)
    plt.title("Indian Dataset")
    plt.xlabel("Time")
    plt.ylabel("Target")

    rawappledata = np.loadtxt(datadirectory + dataset3)
    dataset = rawappledata
    plt.plot(rawappledata)
    plt.grid(True)
    plt.title("Apple Dataset")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.show()
    plt.savefig(save_dir)
    return rawappledata


def differenceTimeSeries(timeseries):
    differdataset = np.zeros(len(timeseries) - 1)
    for i in range(len(differdataset)):
        differdataset[i] = timeseries[i + 1] - timeseries[i]
    return differdataset


def datasetANNcreate(dim, timeseries):
    totallength = len(timeseries)
    totaltraininginstance = totallength - dim
    trainingdataset = np.zeros((totaltraininginstance, dim))
    traininglabels = np.zeros(totaltraininginstance)
    for i in range(totaltraininginstance):
        trainingdataset[i, :] = timeseries[i : i + dim]
        traininglabels[i] = timeseries[i + dim]
    return (trainingdataset, traininglabels)


def datasetANNcreateclass(dim, timeseries):
    totallength = len(timeseries)
    totaltraininginstance = totallength - dim
    trainingdataset = np.zeros((totaltraininginstance, dim))
    traininglabels = np.zeros(totaltraininginstance)
    for i in range(totaltraininginstance):
        trainingdataset[i, :] = timeseries[i : i + dim]
        if timeseries[i + dim] > 0:
            traininglabels[i] = 1
    return (trainingdataset, traininglabels)


def linscale(dataset):
    differdataset = np.zeros(len(dataset))
    for i in range(len(dataset)):
        differdataset[i] = (np.max(dataset) - dataset[i]) / (
            np.max(dataset) - np.min(dataset)
        )

    return differdataset


# Mean absolute percentage error (MAPE) scale invariant error measure 100*1/N* sum over all n (abs(error(n))/true test data(n))
def errorMeasure(prediction, truth):
    error = (
        100 * np.sum(np.abs(prediction - truth) / (truth + 0.0000000001)) / len(truth)
    )
    return error


# Regression
def run_emd(rawappledata, save_dir):
    dataset = differenceTimeSeries(rawappledata)
    emd = EMD()
    imfset = emd.emd(dataset)

    plt.plot(dataset)
    plt.grid(True)
    plt.title("Apple Dataset difference ")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.show()
    plt.savefig(save_dir)

    plt.plot(imfset[9, :])
    plt.grid(True)
    plt.title("Apple Dataset difference ")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.show()
    plt.savefig(save_dir)
    return dataset, imfset


def lstm_prediction(dataset):
    seq = dataset

    input_size = 1
    num_steps = 5
    lstm_size = 128
    num_layers = 1
    batch_size = 100
    max_epoch = 130
    keep_prob = 0.9
    split = 2800

    tf.reset_default_graph()
    lstm_graph = tf.Graph()

    seq = [
        np.array(seq[i * input_size : (i + 1) * input_size])
        for i in range(len(seq) // input_size)
    ]

    # Split into groups of `num_steps`
    X = np.array([seq[i : i + num_steps] for i in range(len(seq) - num_steps)])
    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    x_train = X[0:split, :, :]
    y_train = y[0:split, :]

    x_test = X[split + 1 :, :, :]
    y_test = y[split + 1 :, :]

    total_size = x_train.shape[0]

    with lstm_graph.as_default():
        inputs = tf.placeholder("float", [None, num_steps, input_size])
        targets = tf.placeholder("float", [None, input_size])

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
            if keep_prob < 1.0:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell, output_keep_prob=keep_prob
                )
            return lstm_cell

        cell = (
            tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(num_layers)], state_is_tuple=True
            )
            if num_layers > 1
            else _create_one_cell()
        )

        #    learningrate = 0.1
        val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
        weight = tf.Variable(tf.truncated_normal([lstm_size, input_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[input_size]))
        prediction = tf.matmul(last, weight) + bias
        loss = tf.reduce_mean(tf.square(prediction - targets))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session(graph=lstm_graph) as sess:
        tf.global_variables_initializer().run()
        test_data_feed = {inputs: x_test, targets: y_test}

        for epoch_step in range(max_epoch):
            total_batch = int(total_size / batch_size)

            for i in range(total_batch):
                batch_X = x_train[i * batch_size : (i + 1) * batch_size, :, :]
                batch_y = y_train[i * batch_size : (i + 1) * batch_size, :]
                train_data_feed = {inputs: batch_X, targets: batch_y}
                train_loss, _ = sess.run([optimizer, loss], train_data_feed)

            if epoch_step % 1 == 0:
                (
                    test_loss,
                    _pred,
                ) = sess.run([loss, prediction], test_data_feed)
                train_loss, _ = sess.run([loss, prediction], train_data_feed)
                print("Epoch test loss %d :" % (epoch_step), test_loss)
                print("Epoch train loss %d :" % (epoch_step), train_loss)

        whole_train_data_feed = {inputs: x_train, targets: y_train}
        train_final_prediction, train_final_loss = sess.run(
            [prediction, loss], whole_train_data_feed
        )

        final_prediction, final_loss = sess.run([prediction, loss], test_data_feed)

    y_pred = final_prediction.reshape(-1, 1)
    x_pred = train_final_prediction.reshape(-1, 1)
    return np.squeeze(y_pred), final_loss, np.squeeze(x_pred), train_final_loss


def model_validation(dataset, imfset):
    plainprediction, plain_loss, trainplainprediction, trainplainloss = lstm_prediction(
        dataset
    )
    plain_loss = mean_squared_error(
        plainprediction, dataset[len(dataset) - len(plainprediction) :,]
    )

    binaryupdownlabels = np.zeros(len(plainprediction))
    binaryupdownlabels[dataset[len(dataset) - len(plainprediction) :,] > 0] = 1

    binarypredictionplain = np.zeros(len(plainprediction))
    binarypredictionplain[plainprediction > 0] = 1
    binaryplainaccuracy = np.mean(binarypredictionplain == binaryupdownlabels)
    confmatplain = confusion_matrix(binaryupdownlabels, binarypredictionplain)
    tp = confmatplain[0, 0]
    fp = confmatplain[0, 1]
    fn = confmatplain[1, 0]
    tn = confmatplain[1, 1]
    precisionplain = tp / (tp + fp)
    recallplain = tp / (tp + fn)
    f1plain = 2 * (precisionplain * recallplain) / (precisionplain + recallplain)

    imfpredictionvector = np.zeros((len(imfset), len(plainprediction)))
    imftrainpredictionvector = np.zeros((len(imfset), len(trainplainprediction)))
    imfmeanerrorvector = np.zeros(len(imfset))

    for i in range(len(imfset)):
        #    if i == 9:
        #        break
        print("imf component =" + str(i))
        imfdata = imfset[i, :]
        (
            imfpredictionvector[i, :],
            imfmeanerrorvector[i],
            imftrainpredictionvector[i],
            _,
        ) = lstm_prediction(imfdata)
    print("completed")

    imfcombinedprediction = np.sum(imfpredictionvector, axis=0)
    targetvalues = dataset[len(dataset) - len(plainprediction) :,]
    imfcombinederror = mean_squared_error(imfcombinedprediction, targetvalues)
    print(imfcombinederror)

    binaryupdownlabels = np.zeros(len(plainprediction))
    binaryupdownlabels[dataset[len(dataset) - len(plainprediction) :,] > 0] = 1

    binarypredictionimf = np.zeros(len(plainprediction))
    binarypredictionimf[imfcombinedprediction > 0] = 1
    binaryimfaccuracy = np.mean(binarypredictionimf == binaryupdownlabels)
    confmatimf = confusion_matrix(binaryupdownlabels, binarypredictionimf)
    tp = confmatimf[0, 0]
    fp = confmatimf[0, 1]
    fn = confmatimf[1, 0]
    tn = confmatimf[1, 1]
    precisionimf = tp / (tp + fp)
    recallimf = tp / (tp + fn)
    f1imf = 2 * (precisionimf * recallimf) / (precisionimf + recallimf)


def plot_predictions(
    plot_dir,
    imfcombinedprediction,
    imfpredictionvector,
    plainprediction,
    imftrainpredictionvector,
    dataset,
    targetvalues,
    imfset,
):
    # linear regression combination
    linearregressiondata = np.transpose(imftrainpredictionvector)
    linearregressiontargets = dataset[6 : len(linearregressiondata) + 6]

    regressor = LinearRegression(
        fit_intercept=True, normalize=False, copy_X=True, n_jobs=1
    )
    regressor.fit(linearregressiondata, linearregressiontargets)

    predictedtargets = regressor.predict(linearregressiondata)
    regressorerror = mean_squared_error(predictedtargets, linearregressiontargets)

    predictedtargets = regressor.predict(np.transpose(imfpredictionvector))
    regressorerror = mean_squared_error(predictedtargets, targetvalues)

    totallength = len(dataset)
    testlength = len(plainprediction)
    i = 5
    y = imfset[i, :]
    x = imfpredictionvector[i, :]
    fig = plt.figure()
    plt.grid(True)
    plt.plot(np.arange(0, totallength), y[0:totallength], "g-", label="truth")
    plt.plot(
        np.arange(len(dataset) - len(plainprediction), totallength),
        x,
        "r--",
        label="Prediction",
    )
    plt.plot(
        np.arange(6, len(linearregressiondata) + 6),
        imftrainpredictionvector[i, :],
        "b-",
        label="Prediction",
    )

    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title(" RNN prediction imf component =" + str(i))
    plt.legend(loc="upper left")
    plt.xlim([2700, 3088])
    plt.savefig(
        plot_dir + str(i) + ".png",
        dpi=fig.dpi,
    )

    np.sum(imftrainpredictionvector, axis=0)
    y = dataset
    x = imfcombinedprediction
    fig = plt.figure()
    plt.grid(True)
    plt.plot(np.arange(2700, totallength), y[2700:totallength], "g-", label="truth")
    plt.plot(
        np.arange(len(dataset) - len(plainprediction), totallength),
        x,
        "r--",
        label="Prediction",
    )
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title("RNN Combined Prediction")
    plt.legend(loc="upper left")
    plt.xlim([2700, 3088])
    plt.savefig(
        plot_dir,
        dpi=fig.dpi,
    )

    y = dataset
    x = plainprediction
    fig = plt.figure()
    plt.grid(True)
    plt.plot(np.arange(2700, totallength), y[2700:totallength], "g-", label="truth")
    plt.plot(
        np.arange(len(dataset) - len(plainprediction), totallength),
        x,
        "r--",
        label="Prediction",
    )
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title("RNN plain prediction")
    plt.legend(loc="upper left")
    plt.xlim([2700, 3088])
    plt.savefig(
        plot_dir,
        dpi=fig.dpi,
    )


def main():
    datadirectory = "data_dir"
    save_dir = "save_dir"
    rawappledata = read_data(datadirectory, save_dir)
    dataset, imfset = run_emd(rawappledata, save_dir)
    model_validation(dataset, imfset)