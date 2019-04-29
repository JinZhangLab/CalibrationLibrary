#
import scipy.io as scio
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from calib.pls import MCuve


if __name__ == "__main__":
    MatPath = './data/corn.mat'
    CornData = scio.loadmat(MatPath)
    wv = np.linspace(1100, 2498, (2498-1100)//2+1, endpoint=True)
    X = CornData["X"]
    Y = CornData["Y"]
    print("Importted the dataset")

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y[:, 0], test_size=0.2)
    plsmodel = PLSRegression(n_components=7)
    plsmodel.fit(Xtrain, Ytrain)
    T, P, U, Q, W, C, beta = plsmodel.x_scores_, plsmodel.x_loadings_, plsmodel.y_scores_, plsmodel.y_loadings_, plsmodel.x_weights_, plsmodel.y_weights_, plsmodel.coef_
    plt.plot(wv, beta[0:])
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Regression Coefficients")
    plt.show()

    mcmodel = MCuve(Xtrain, Ytrain, 7)
    stability = mcmodel.calcStability()
    plt.plot(stability)
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Stability of MCUVE")
    plt.show()