import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

class MCuve:
    def __init__(self, x, y, nComp = 1, nrep = 500, testSize = 0.2):
        self.x = x
        self.y = y
        self.ncomp = nComp
        self.nrep = nrep
        self.testSize = testSize

    def calcStability(self):
        stability = np.zeros(self.x.shape[1])
        PLSCoef = np.zeros((self.nrep, self.x.shape[1]))
        ss = ShuffleSplit(n_splits = self.nrep, test_size = self.testSize)
        step = 0
        for train, test in ss.split(self.x, self.y):
            xtrain = self.x[train,:]
            ytrain = self.y[train]
            plsModel = PLSRegression(self.ncomp)
            plsModel.fit(xtrain, ytrain)
            PLSCoef[step, :] = plsModel.coef_.T
            step += 1
        meanCoef = np.mean(PLSCoef, axis=0)
        stdCoef = np.std(PLSCoef, axis=0)
        stability = meanCoef/stdCoef
        return stability


if __name__ == "__main__":
    print("This is the PLS model")