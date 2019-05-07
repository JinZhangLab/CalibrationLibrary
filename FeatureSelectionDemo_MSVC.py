#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019.4.30
# @Author  : FrankEl
# @File    : Feature_selection_demo_rt.py

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from numpy.linalg import matrix_rank as rank
from calib.FeatureSelection import MSVC

if __name__ == "__main__":
    ncomp = 7
    MatPath = './data/corn.mat'
    CornData = scio.loadmat(MatPath)
    wv = np.linspace(1100, 2498, (2498 - 1100) // 2 + 1, endpoint=True)
    X = CornData["X"]
    Y = CornData["Y"]
    # Building normal pls model
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y[:, 0], test_size=0.2)
    plsModel = PLSRegression(n_components=ncomp)
    plsModel.fit(Xtrain, Ytrain)
    T, P, U, Q, W, C, beta = plsModel.x_scores_, plsModel.x_loadings_, plsModel.y_scores_, plsModel.y_loadings_, plsModel.x_weights_, plsModel.y_weights_, plsModel.coef_
    plt.plot(wv, beta[0:])
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Regression Coefficients")
    plt.savefig("./Image/Image1_RegressionCoefficient_PLS.png")
    plt.close()

    # Prediction result of pls model
    Ytrain_hat = plsModel.predict(Xtrain)
    Ytest_hat = plsModel.predict(Xtest)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, Ytrain_hat, marker='*')
    plt.scatter(Ytest, Ytest_hat, marker='*')
    plt.xlabel("Prediction")
    plt.ylabel("Reference")
    plt.title("Prediction of normal pls model")
    plt.savefig("./Image/Image2_PredictionPLS.png")
    plt.close()

    # C value of MSVC
    vcModel = MSVC(Xtrain, Ytrain, ncomp, nrep=500)
    vcModel.calcCriteria()
    plt.imshow(vcModel.criteria, aspect='auto')
    plt.xlabel("Wavelength")
    plt.ylabel("Iteration")
    plt.title("C value of VC")
    plt.savefig("./Image/Image3_MS_C_vale.png")
    plt.close()

    # Feature ranking efficienty by stability of VC
    vcModel.evalCriteria(cv=3)
    plt.plot(vcModel.featureR2)
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("R2")
    plt.savefig("./Image/Image4_R2.png")
    plt.close()

    # Prediction results after feature selection by VC
    XtrainNew, XtestNew = vcModel.cutFeature(Xtrain, Xtest)
    plsModelNew = PLSRegression(n_components=min([ncomp, rank(XtrainNew)]))
    plsModelNew.fit(XtrainNew, Ytrain)
    YtrainNew_hat = plsModelNew.predict(XtrainNew)
    YtestNew_hat = plsModelNew.predict(XtestNew)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, YtrainNew_hat, marker='*')
    plt.scatter(Ytest, YtestNew_hat, marker='*')
    plt.xlabel("Prediction")
    plt.ylabel("Reference")
    plt.title("Prediction after MSVC")
    plt.savefig("./Image/Image5_Prediction_MSVC.png")
    plt.close()