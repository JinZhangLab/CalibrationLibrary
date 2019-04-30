#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019.4.30
# @Author  : FrankEl
# @File    : Feature_selection_demo_mcuve.py

import scipy.io as scio
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from calib.pls import MCuve

if __name__ == "__main__":
    MatPath = './data/corn.mat'
    CornData = scio.loadmat(MatPath)
    wv = np.linspace(1100, 2498, (2498 - 1100) // 2 + 1, endpoint=True)
    X = CornData["X"]
    Y = CornData["Y"]
    # Building normal pls model
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y[:, 0], test_size=0.2)
    plsModel = PLSRegression(n_components=7)
    plsModel.fit(Xtrain, Ytrain)
    T, P, U, Q, W, C, beta = plsModel.x_scores_, plsModel.x_loadings_, plsModel.y_scores_, plsModel.y_loadings_, plsModel.x_weights_, plsModel.y_weights_, plsModel.coef_
    plt.plot(wv, beta[0:])
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Regression Coefficients")
    plt.show()
    # Prediction result of pls model
    Ytrain_hat = plsModel.predict(Xtrain)
    Ytest_hat = plsModel.predict(Xtest)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, Ytrain_hat, marker='*')
    plt.scatter(Ytest, Ytest_hat, marker='*')
    plt.xlabel("Reference")
    plt.ylabel("Prediction")
    plt.title("Prediction of normal pls model")
    plt.show()
    # Stability of MC-UVE
    mcModel = MCuve(Xtrain, Ytrain, 7)
    mcModel.calcStability()
    plt.plot(mcModel.stability)
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Stability of MCUVE")
    plt.show()
    # Feature ranking efficienty by stability of MC-UVE
    mcModel.evalStability(cv=5)
    plt.plot(mcModel.featureR2)
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("R2")
    plt.show()
    # Prediction results after feature selection by MC-UVE
    XtrainNew, XtestNew = mcModel.cutFeature(Xtrain, Xtest)
    plsModelNew = PLSRegression(n_components=7)
    plsModelNew.fit(XtrainNew, Ytrain)
    YtrainNew_hat = plsModelNew.predict(XtrainNew)
    YtestNew_hat = plsModelNew.predict(XtestNew)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, YtrainNew_hat, marker='*')
    plt.scatter(Ytest, YtestNew_hat, marker='*')
    plt.xlabel("Reference")
    plt.ylabel("Prediction")
    plt.title("Prediction after MC-UVE")
    plt.show()