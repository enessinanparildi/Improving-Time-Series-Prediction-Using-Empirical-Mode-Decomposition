# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:59:36 2017

@author: Ben
"""

from PyEMD import EMD
import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    WhiteKernel,
    RationalQuadratic,
    ExpSineSquared,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from pywt import wavedec
from sklearn.metrics import confusion_matrix

datadirectory = "C:\\Users\\Ben\\Google Drive\\masterders\\advanced signal processing\\project\\datasets\\"
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
        trainingdataset[i, :] = timeseries[i: i + dim]
        traininglabels[i] = timeseries[i + dim]
    return (trainingdataset, traininglabels)


def datasetANNcreateclass(dim, timeseries):
    totallength = len(timeseries)
    totaltraininginstance = totallength - dim
    trainingdataset = np.zeros((totaltraininginstance, dim))
    traininglabels = np.zeros(totaltraininginstance)
    for i in range(totaltraininginstance):
        trainingdataset[i, :] = timeseries[i: i + dim]
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


rawdataset = rawappledata
totallength = len(rawdataset)
trainlength = 2805
testlength = totallength - trainlength

# Regression
dataset = differenceTimeSeries(rawappledata)
# dataset = linscale(rawappledata)
dim = 5
(trainingdataset, traininglabels) = datasetANNcreate(dim, dataset[0:trainlength])
(testdataset, testlabels) = datasetANNcreate(
    dim, dataset[trainlength - dim: totallength]
)
originaltestlabels = testlabels
originaltrainingset = trainingdataset
originaltraininglabels = traininglabels
originaltestdataset = testdataset

binaryupdownlabels = np.zeros(len(originaltestdataset))
binaryupdownlabels[originaltestlabels > 0] = 1


def getErr(predictions):
    binpredictions = np.zeros(len(predictions))
    binpredictions[prediction > 0] = 1
    binarygpaccuracy = np.mean(binpredictions == binaryupdownlabels)

    confmatgp = confusion_matrix(binaryupdownlabels, binarygpaccuracy)
    tp = confmatgp[0, 0]
    fp = confmatgp[0, 1]
    fn = confmatgp[0, 1]
    tn = confmatgp[1, 1]
    precisiongp = tp / (tp + fp)
    recallrgp = tp / (tp + fn)
    f1gp = 2 * (precisiongp * recallrgp) / (precisiongp + recallrgp)
    return (f1gp, precisiongp, recallrgp, binarygpaccuracy)


plt.plot(dataset)
plt.grid(True)
plt.title("Apple Dataset Difference")
plt.xlabel("Time")
plt.ylabel("Target")
plt.show()


def neuralNetworkPredictionRegression(
        trainingdataset,
        testdataset,
        traininglabels,
        testlabels,
        betanum=0.00001,
        epochnum=100,
):
    x = tf.placeholder("float", [None, dim])
    y = tf.placeholder("float", [None])

    n_hidden_1 = 20
    n_hidden_2 = 20
    n_hidden_3 = 20

   class MLP(tf.keras.Model):
        def __init__(self):
            super(MLP, self).__init__()
    
            # Hidden layer 1 with tanh activation
            self.layer1 = tf.keras.layers.Dense(
                n_hidden_1,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=math.sqrt(2.0 / dim)
                ),
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.l2(betanum),
            )
    
            # Hidden layer 2 with tanh activation
            self.layer2 = tf.keras.layers.Dense(
                n_hidden_2,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=math.sqrt(2.0 / n_hidden_1)
                ),
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.l2(betanum),
            )
    
            # Hidden layer 3 with tanh activation
            self.layer3 = tf.keras.layers.Dense(
                n_hidden_3,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=math.sqrt(2.0 / n_hidden_2)
                ),
                bias_initializer="zeros",
                kernel_regularizer=tf.keras.regularizers.l2(betanum),
            )
    
            # Output layer with linear activation
            self.out_layer = tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=math.sqrt(2.0 / n_hidden_3)
                ),
                bias_initializer="zeros",
            )
    
        def call(self, x, training=False):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.out_layer(x)
            return tf.squeeze(x, axis=-1)


    model = MLP()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=epochnum, batch_size=batch_size, verbose=1)
    
    
    test_predictions = model.predict(testdataset)
    test_mse_error_vector = tf.keras.losses.mean_squared_error(testlabels, test_predictions)
    mean_error = np.mean(test_mse_error_vector)
    
    # Calculate predictions and loss for training data
    train_final_prediction = model.predict(trainingdataset)
    train_final_loss = model.evaluate(trainingdataset, traininglabels, verbose=0)
    
    percantageerror = errorMeasure(predictions, testlabels)
    
    return (predictions, mean_error, train_final_prediction, train_final_loss)
    

def fnn_regression():
    (
        predictionsplain,
        mean_vectorplain,
        trainplainprediction,
        trainplainloss,
    ) = neuralNetworkPredictionRegression(
        originaltrainingset, originaltestdataset, originaltraininglabels, originaltestlabels
    )

    binarypredictionplain = np.zeros(len(originaltestdataset))
    binarypredictionplain[predictionsplain > 0] = 1
    binaryplainaccuracy = np.mean(binarypredictionplain == binaryupdownlabels)
    confmatplain = confusion_matrix(binaryupdownlabels, binarypredictionplain)
    tp = confmatplain[0, 0]
    fp = confmatplain[0, 1]
    fn = confmatplain[1, 0]
    tn = confmatplain[1, 1]
    precisionplain = tp / (tp + fp)
    recallplain = tp / (tp + fn)
    f1plain = 2 * (precisionplain * recallplain) / (precisionplain + recallplain)
    # np.mean(predictionsplain)
    # np.mean(originaltestlabels)


emd = EMD()
imfset = emd.emd(dataset)

imfpredictionvector = np.zeros((len(imfset), len(testlabels)))
imftrainpredictionvector = np.zeros((len(imfset), len(traininglabels)))
imfmeanerrorvector = np.zeros(len(imfset))
io.savemat(
    "C:\\Users\\Ben\\Google Drive\\masterders\\advanced signal processing\\project\\datasets\\diffappleimfcomps.mat",
    mdict={"arr": imfset},
)
# EMD prediction
for i in range(len(imfset)):
    #    if i == 1:
    #        break
    print("imf component =" + str(i))
    imfdata = imfset[i, :]
    (trainingdataset, traininglabels) = datasetANNcreate(dim, imfdata[0:trainlength])
    (testdataset, testlabels) = datasetANNcreate(
        dim, imfdata[trainlength - dim: totallength]
    )
    (
        imfpredictionvector[i, :],
        imfmeanerrorvector[i],
        imftrainpredictionvector[i],
        _,
    ) = neuralNetworkPredictionRegression(
        trainingdataset, testdataset, traininglabels, testlabels
    )

imfcombinedprediction = np.sum(imfpredictionvector, axis=0)
imfcombinederror = mean_squared_error(imfcombinedprediction, originaltestlabels)
print(imfcombinederror)

binarypredictionimf = np.zeros(len(originaltestdataset))
binarypredictionimf[predictedtargets > 0] = 1
binaryimfaccuracy = np.mean(binarypredictionimf == binaryupdownlabels)
print(binaryimfaccuracy)
confmatimf = confusion_matrix(binaryupdownlabels, binarypredictionimf)
tp = confmatimf[0, 0]
fp = confmatimf[0, 1]
fn = confmatimf[1, 0]
tn = confmatimf[1, 1]
precisionimf = tp / (tp + fp)
recallimf = tp / (tp + fn)
f1imf = 2 * (precisionimf * recallimf) / (precisionimf + recallimf)

# linear regression combination full training data
linearregressiondata = np.transpose(imfpredictionvector)
linearregressiontargets = originaltestlabels

regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
regressor.fit(linearregressiondata, linearregressiontargets)

predictedtargets = regressor.predict(linearregressiondata)
regressorerror = mean_squared_error(predictedtargets, linearregressiontargets)
#
# predictedtargets = regressor.predict(np.transpose(imfpredictionvector))
# regressorerror = mean_squared_error(predictedtargets, originaltestlabels)


# linear regression combination part of the test
linearregressiondata = np.transpose(imfpredictionvector[:, :200])
linearregressiontargets = originaltestlabels[:200]

regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
regressor.fit(linearregressiondata, linearregressiontargets)

predictedtargets = regressor.predict(np.transpose(imfpredictionvector[:, 200:]))
regressorerror = mean_squared_error(predictedtargets, originaltestlabels[200:])

predictedtargets = regressor.predict(np.transpose(imfpredictionvector))
regressorerror = mean_squared_error(
    np.sum(imfpredictionvector[:, 200:], axis=0), originaltestlabels[200:]
)


def run_svm_model():
    # Support vector regression
    svr_rbf = SVR(kernel="rbf", C=16, gamma=0.001, epsilon=0.2)
    svr_lin = SVR(kernel="linear", C=16, epsilon=0.2)
    svr_poly = SVR(kernel="poly", C=16, degree=2, epsilon=0.2)

    y_rbf = svr_rbf.fit(originaltrainingset, originaltraininglabels).predict(
        originaltestdataset
    )
    y_lin = svr_lin.fit(originaltrainingset, originaltraininglabels).predict(
        originaltestdataset
    )
    y_poly = svr_poly.fit(originaltrainingset, originaltraininglabels).predict(
        originaltestdataset
    )

    rbferror = mean_squared_error(y_rbf, originaltestlabels)
    linerror = mean_squared_error(y_lin, originaltestlabels)
    polyerror = mean_squared_error(y_poly, originaltestlabels)

    binarypredictionrbf = np.zeros(len(originaltestdataset))
    binarypredictionrbf[y_rbf > 0] = 1
    binaryrbfaccuracy = np.mean(binarypredictionrbf == binaryupdownlabels)
    print(binaryrbfaccuracy)

    binarypredictionrbf = np.zeros(len(originaltestdataset))
    binarypredictionrbf[y_rbf > 0] = 1
    confmatrbf = confusion_matrix(binaryupdownlabels, binarypredictionimf)
    tp = confmatrbf[0, 0]
    fp = confmatrbf[0, 1]
    fn = confmatrbf[1, 0]
    tn = confmatrbf[1, 1]
    precisionrbf = tp / (tp + fp)
    recallrbf = tp / (tp + fn)
    f1rbf = 2 * (precisionrbf * recallrbf) / (precisionrbf + recallrbf)

    binarypredictionimf = np.zeros(len(originaltestdataset))
    binarypredictionimf[y_lin > 0] = 1
    binaryimfaccuracy = np.mean(binarypredictionimf == binaryupdownlabels)
    print(binaryimfaccuracy)
    confmatimf = confusion_matrix(binaryupdownlabels, binarypredictionimf)
    tp = confmatimf[0, 0]
    fp = confmatimf[0, 1]
    fn = confmatimf[1, 0]
    tn = confmatimf[1, 1]
    precisionimf = tp / (tp + fp)
    recallimf = tp / (tp + fn)
    f1imf = 2 * (precisionimf * recallimf) / (precisionimf + recallimf)

    emd = EMD()
    imfset = emd.emd(dataset)
    rbfimfpredictionvector = np.zeros((len(imfset), len(originaltestdataset)))
    linimfpredictionvector = np.zeros((len(imfset), len(originaltestdataset)))
    polyimfpredictionvector = np.zeros((len(imfset), len(originaltestdataset)))
    imfmeanerrorvector = np.zeros(len(imfset))
    # EMD prediction
    for i in range(len(imfset)):
        #    if i == 1:
        #        break
        print("imf component =" + str(i))
        imfdata = imfset[i, :]
        (trainingdataset, traininglabels) = datasetANNcreate(dim, imfdata[0:trainlength])
        (testdataset, testlabels) = datasetANNcreate(
            dim, imfdata[trainlength - dim: totallength]
        )

        rbfimfpredictionvector[i, :] = svr_rbf.fit(trainingdataset, traininglabels).predict(
            testdataset
        )
        linimfpredictionvector[i, :] = svr_lin.fit(trainingdataset, traininglabels).predict(
            testdataset
        )
        polyimfpredictionvector[i, :] = svr_poly.fit(
            trainingdataset, traininglabels
        ).predict(testdataset)

    imfrbfcombinedprediction = np.sum(rbfimfpredictionvector, axis=0)
    imflincombinedprediction = np.sum(linimfpredictionvector, axis=0)
    imfpolycombinedprediction = np.sum(polyimfpredictionvector, axis=0)

    imfrbfcombinederror = mean_squared_error(imfrbfcombinedprediction, originaltestlabels)
    imflincombinederror = mean_squared_error(imflincombinedprediction, originaltestlabels)
    imfpolycombinederror = mean_squared_error(imfpolycombinedprediction, originaltestlabels)

    print(imflincombinederror)
    binarypredictionlin = np.zeros(len(originaltestdataset))
    binarypredictionlin[imflincombinedprediction > 0] = 1
    binarylinaccuracy = np.mean(binarypredictionlin == binaryupdownlabels)

    confmatlin = confusion_matrix(binaryupdownlabels, binarypredictionlin)
    tp = confmatlin[0, 0]
    fp = confmatlin[0, 1]
    fn = confmatlin[1, 0]
    tn = confmatlin[1, 1]
    precisionlin = tp / (tp + fp)
    recalllin = tp / (tp + fn)
    f1lin = 2 * (precisionlin * recalllin) / (precisionlin + recalllin)

    binarypredictionrbf = np.zeros(len(originaltestdataset))
    binarypredictionrbf[imfrbfcombinedprediction > 0] = 1
    binaryrbfaccuracy = np.mean(binarypredictionrbf == binaryupdownlabels)

    confmatrbf = confusion_matrix(binaryupdownlabels, binarypredictionrbf)
    tp = confmatrbf[0, 0]
    fp = confmatrbf[0, 1]
    fn = confmatrbf[1, 0]
    tn = confmatrbf[1, 1]
    precisionrbf = tp / (tp + fp)
    recallrbf = tp / (tp + fn)
    f1rbf = 2 * (precisionrbf * recallrbf) / (precisionrbf + recallrbf)

# gaussian process regression
## raw data

def run_gaussian_process_regression(rawdataset):

    dataset = rawdataset
    dataset = differenceTimeSeries(rawdataset)
    emd = EMD()
    imfset = emd.emd(dataset)
    dim = 2
    totallength = len(rawdataset)
    trainlength = 2803
    testlength = totallength - trainlength

    (trainingdataset, traininglabels) = datasetANNcreate(dim, dataset[0:trainlength])
    (testdataset, originaltestlabels) = datasetANNcreate(
        dim, dataset[trainlength:totallength]
    )

    kernel = 0.01 * RBF(length_scale=10000, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=10, noise_level_bounds=(1e-10, 1e1)
    )
    kernel = Matern() + RBF()

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    #
    gp.fit(trainingdataset, traininglabels.reshape(-1, 1))
    prediction, sigma = gp.predict(testdataset, return_std=True)

    gperrorraw = mean_squared_error(prediction, originaltestlabels)

    binarypredictionrbf = np.zeros(len(prediction))
    binarypredictionrbf[np.squeeze((np.transpose(prediction))) > 0] = 1
    binaryrbfaccuracy = np.mean(binarypredictionrbf == binaryupdownlabels)

    confmatrbf = confusion_matrix(binaryupdownlabels, binarypredictionrbf)
    tp = confmatrbf[0, 0]
    fp = confmatrbf[0, 1]
    fn = confmatrbf[1, 0]
    tn = confmatrbf[1, 1]
    precisionrbf = tp / (tp + fp)
    recallrbf = tp / (tp + fn)
    f1rbf = 2 * (precisionrbf * recallrbf) / (precisionrbf + recallrbf)

    gppredictionvector = np.zeros((len(imfset), len(originaltestlabels)))
    gpmeanerrorvector = np.zeros(len(imfset))
    for i in range(len(imfset)):
        print("imf component =" + str(i))
        dataset = imfset[i, :]
        (trainingdataset, traininglabels) = datasetANNcreate(dim, dataset[0:trainlength])
        (testdataset, testlabels) = datasetANNcreate(dim, dataset[trainlength:totallength])

        gp.fit(trainingdataset, traininglabels)
        gppredictionvector[i, :], _ = gp.predict(testdataset, return_std=True)
        gpmeanerrorvector[i] = mean_squared_error(gppredictionvector[i, :], testlabels)

    gpcombinedprediction = np.sum(gppredictionvector, axis=0)
    gpimfcombinederror = mean_squared_error(gpcombinedprediction, originaltestlabels)

    binarypredictionrbf = np.zeros(len(prediction))
    binarypredictionrbf[gpcombinedprediction > 0] = 1
    binaryrbfaccuracy = np.mean(binarypredictionrbf == binaryupdownlabels)

    confmatrbf = confusion_matrix(binaryupdownlabels, binarypredictionrbf)
    tp = confmatrbf[0, 0]
    fp = confmatrbf[0, 1]
    fn = confmatrbf[1, 0]
    tn = confmatrbf[1, 1]
    precisionrbf = tp / (tp + fp)
    recallrbf = tp / (tp + fn)
    f1rbf = 2 * (precisionrbf * recallrbf) / (precisionrbf + recallrbf)

# Discrete wavelet transform


# ARIMA model raw dataset
dataset = rawdataset
totallength = len(rawdataset)
trainlength = 2805
testlength = totallength - trainlength
trainingdata = dataset[0:trainlength]
testdata = dataset[trainlength:totallength]

model = ARMA(trainingdata, order=(5, 1))
model_fit = model.fit(trend="nc", disp=0)
arimaplainpredictions, stderr, conf_int = model_fit.forecast(len(testdata))
rawplainerror = mean_squared_error(testdata, arimaplainpredictions)
# 285 is found now imf

emd = EMD()
imfset = emd.emd(dataset)
io.savemat(
    "C:\\Users\\Ben\\Google Drive\\masterders\\advanced signal processing\\project\\datasets\\rawappleimfcomps.mat",
    mdict={"arr": imfset},
)
imfpredictionvector = np.zeros((len(imfset), len(arimaplainpredictions)))
imfmeanerrorvector = np.zeros(len(imfset))
for i in range(1, len(imfset)):
    print("imf component =" + str(i))
    imfdata = imfset[i, :]
    trainingdata = imfdata[0:trainlength]
    testdata = imfdata[trainlength:totallength]
    model = ARMA(trainingdata, order=(2, 0))
    model_fit = model.fit(transparams=False)
    imfpredictionvector[i, :], stderr, conf_int = model_fit.forecast(len(testdata))

imfcombinedprediction = np.sum(imfpredictionvector, axis=0)
arimaimfcombinederror = mean_squared_error(imfcombinedprediction, testdata)

plt.plot(imfset[9, :])
plt.grid(True)
plt.title("Apple Dataset")
plt.xlabel("Time")
plt.ylabel("Target")

#
# k = gpflow.kernels.Matern52(1, lengthscales=0.3)
# m = gpflow.models.GPR(X, Y, kern=k)
# m.likelihood.variance = 0.01
#


y_pred = sampledprediction
X = dummytrainingdata
x = dummytestdata
y = trainingdata

fig = plt.figure()
plt.grid(True)
plt.plot(x, testdata, "g.", markersize=10, label="truth")
plt.plot(X, y, "r.", markersize=10, label="Observations")
plt.plot(x, y_pred, "b-", label="Prediction")
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend(loc="upper left")

emd = EMD()
imfset = emd.emd(dataset)
imfpredictionvector = np.zeros((len(imfset), testlength))
imfmeanerrorvector = np.zeros(len(imfset))
for i in range(1, len(imfset)):
    print("imf component =" + str(i))
    imfdata = imfset[i, :]
    trainingdata = imfdata[0:trainlength]

    gp.fit(np.arange(len(trainingdata)).reshape(-1, 1), trainingdata)
    prediction = gp.predict(np.arange(len(trainingdata), totallength).reshape(-1, 1))
    imfpredictionvector[i, :] = prediction

imfcombinedprediction = np.sum(imfpredictionvector, axis=0)
gpimfcombinederror = mean_squared_error(imfcombinedprediction, testdata)

# classification
dataset = differenceTimeSeries(rawdataset)
dim = 15
(trainingdataset, traininglabels) = datasetANNcreateclass(dim, dataset[0:trainlength])
(testdataset, testlabels) = datasetANNcreateclass(dim, dataset[trainlength:totallength])
originaltestlabels = testlabels
originaltrainingset = trainingdataset
originaltraininglabels = traininglabels
originaltestdataset = testdataset


def fnn_classification(trainingdataset, testdataset, traininglabels, testlabels):
    traininglabels = traininglabels.reshape((len(traininglabels), 1))
    testlabels = testlabels.reshape((len(testlabels), 1))

    onehottest = np.zeros((len(testlabels), 2))
    onehottrain = np.zeros((len(traininglabels), 2))
    for i in range(len(testlabels)):
        onehottest[i, int(testlabels[i])] = 1
    for i in range(len(traininglabels)):
        onehottrain[i, int(traininglabels[i])] = 1

    traininglabels = onehottrain
    testlabels = onehottest

    x = tf.placeholder("float", [None, dim])
    y = tf.placeholder("float", [None, 2])

    n_hidden_1 = 30
    n_hidden_2 = 20
    n_hidden_3 = 10

    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.tanh(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.tanh(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
        layer_3 = tf.nn.tanh(layer_3)

        out_layer = tf.matmul(layer_3, weights["out"]) + biases["out"]
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    weights = {
        "h1": tf.Variable(
            tf.truncated_normal([dim, n_hidden_1], stddev=math.sqrt(2.0 / (dim)))
        ),
        "h2": tf.Variable(
            tf.truncated_normal(
                [n_hidden_1, n_hidden_2], stddev=math.sqrt(2.0 / (n_hidden_1))
            )
        ),
        "h3": tf.Variable(
            tf.truncated_normal(
                [n_hidden_2, n_hidden_3], stddev=math.sqrt(2.0 / (n_hidden_2))
            )
        ),
        "out": tf.Variable(
            tf.truncated_normal([n_hidden_3, 2], stddev=math.sqrt(2.0 / (n_hidden_3)))
        ),
    }
    biases = {
        "b1": tf.Variable(tf.zeros([n_hidden_1])),
        "b2": tf.Variable(tf.zeros([n_hidden_2])),
        "b3": tf.Variable(tf.zeros([n_hidden_3])),
        "out": tf.Variable(tf.zeros([2])),
    }

    pred = multilayer_perceptron(x, weights, biases)
    #    pred = tf.squeeze(pred)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    init = tf.global_variables_initializer()
    training_epochs = 100
    batch_size = 28
    display_step = 1
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(traininglabels) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = trainingdataset[i * batch_size: (i + 1) * batch_size, :]
                batch_y = traininglabels[i * batch_size: (i + 1) * batch_size]
                #                    batch_y = np.transpose(batch_y)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                #                print("ber:" , ber )
                print(
                    "Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
                )
        print("Optimization Finished!")
        #        saver.save(sess,trainedmodeldirectory)
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        predictions = tf.argmax(pred, 1)
        predictions = predictions.eval(feed_dict={x: testdataset})
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = accuracy.eval(feed_dict={x: testdataset, y: testlabels})
    return (predictions, accuracy)


def get_results():
    predictionsplain, accuracyplain = fnn_classification(
        originaltrainingset,
        originaltestdataset,
        originaltraininglabels,
        originaltestlabels,
    )

    emd = EMD()
    imfset = emd.emd(dataset)
    imfpredictionvector = np.zeros((len(imfset), len(predictionsplain)))
    imfaccuracyvector = np.zeros(len(imfset))

    # EMD prediction
    for i in range(len(imfset)):
        print("imf component =" + str(i))
        imfdata = imfset[i, :]
        (trainingdataset, traininglabels) = datasetANNcreateclass(
            dim, imfdata[0:trainlength]
        )
        (testdataset, testlabels) = datasetANNcreateclass(
            dim, imfdata[trainlength:totallength]
        )
        imfpredictionvector[i, :], imfaccuracyvector[i] = fnn_classification(
            trainingdataset, testdataset, traininglabels, testlabels
        )

    imfcombinedaccuracy = np.mean(imfaccuracyvector)
    return imfcombinedaccuracy
