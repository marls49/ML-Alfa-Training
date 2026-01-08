import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)


#define Klasse
class HebbLernen:
    def __init__(self,eta=0.01, tmax=1000):
        self.eta = eta
        self.tmax = tmax
        self.w = None

    def myHeaviside(self, x):
        y = np.ones_like(x,dtype=float)
        y[x <= 0] = 0 
        return(y)
        
    def fit (self,x,y):
        self.w = np.random.rand(3) - 0.5
        t = 0 
        convergenz = 1
        while (convergenz > 0) and (t<self.tmax): 
            t += 1
            randomInt = np.random.randint(len(y))
            xB = np.ones(3)
            xB[0:2] = x[randomInt, :]
            yB = y[randomInt]
            error = yB - self.myHeaviside(self.w@xB)
            Dw = np.zeros(3)
            for j in range(len(xB)):
                Dw[j]= self.eta*error*xB[j]
                self.w[j] += Dw[j]
            xC = np.ones((x.shape[0], 3))
            xC[:, 0:2] = x
            convergenz = np.linalg.norm(
                y - self.myHeaviside(self.w @ xC.T)
            )


    def predict(self, x, xMin, xMax):
        xC = np.ones( (x.shape[0],3) )
        xC[:,0:2] = x
        xC[:,0:2] = (xC[:,0:2] - xMin) / (xMax - xMin); print(xC)
        y = self.w@xC.T
        y[y>0] = 1
        y[y<= 0] = 0
        return(y)
  
#define main
if __name__ == "__main__":
    dataset = np.loadtxt("/Users/mileslynam-smith/Downloads/BuchCode/Kapitel7/Autoklassifizierung.csv", delimiter=",")
    y = dataset[:,0]
    X = dataset[:,0:2]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X, y=None)  
    model = HebbLernen(eta=0.25, tmax=100000)
    model.fit(X_scaled, y)
    model.xMin = scaler.data_min_
    model.xMax = scaler.data_max_

    xTest = np.array([[12490, 48], [31590, 169],[24740, 97], [30800, 156]])
    yPredict = model.predict(xTest, model.xMin, model.xMax)
    print(yPredict)



