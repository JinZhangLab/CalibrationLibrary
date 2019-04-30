import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

class MCuve:
    def __init__(self, x, y, nComp = 1, nrep = 500, testSize = 0.2):
        self.x = x
        self.y = y
        self.ncomp = nComp
        self.nrep = nrep
        self.testSize = testSize
        self.stability = None
        self.featureIndex = None
        self.featureR2 = np.empty(self.x.shape[1])
        self.selFeature = None

    def calcStability(self):
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
        self.stability = meanCoef/stdCoef

    def evalStability(self, cv=3, n_jobs=None):
        self.featureIndex = np.argsort(-np.abs(self.stability))
        for i in range(self.x.shape[1]):
            if i<self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(self.ncomp)
            xi = self.x[:, self.featureIndex[:i+1]]
            cvScore = cross_val_score(regModel, xi, self.y, cv=cv)
            self.featureR2[i] = np.mean(cvScore)

    def cutFeature(self, *args):
        cuti = np.argmax(self.featureR2)
        self.selFeature = self.featureIndex[:cuti+1]
        if len(args) != 0:
            returnx = list(args)
            i = 0
            for argi in args:
                if argi.shape[1] == self.x.shape[1]:
                    returnx[i] = argi[:, self.selFeature]
                i += 1
        return tuple(returnx)

if __name__ == "__main__":
    print("This is the PLS model")