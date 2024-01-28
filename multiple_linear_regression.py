import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ref:https://medium.com/analytics-vidhya/multiple-linear-regression-from-scratch-using-python-db9368859f
class Multiple_LinReg():
    def __init__(self,learning_rate= 0.01, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def printResult(self):
        weight_list = self.weights.T.tolist()
        result = pd.DataFrame(weight_list,columns=["distance","precedingAcceleration","precedingSpeed","followingSpeed"])
        print(result)
    
    # don't include y-intercept 
    def fit(self,X,y):
        self.X = X
        self.y = y
        # for including y-intercept
        # X = np.cat_[np.ones((X.shape[0],1)),X]
        self.weights = np.ones((X.shape[1],1))
        for i in range(self.epochs):
            pred_y = np.dot(X,self.weights)
            partial = (2/y.shape[0])*(np.sum(np.dot(X.T,(pred_y-y.reshape(-1, 1))),axis=1,keepdims=True))
            self.weights = self.weights -self.learning_rate*partial
            print_progessbar(self.epochs,i+1)
        self.printResult()
        print("MSE is" ,self.calculateError(pred_y))
    
    # predict value
    def predict(self,X):
        z = np.dot(X,self.weights)
        return z
    
    def illustrateResult(self):
        NotImplementedError

    def calculateError(self,pred_y):
        mse = (1/self.y.shape[0]) *np.sum((pred_y-self.y.reshape(-1, 1))**2)
        return mse

        
def print_progessbar(total,current,barsize = 60):
    progress = int(current*barsize/total)   # stardardize current into bar
    completed = str(int(current/total*100))+"%"      # showing progress how many percent
    print_frequency = max(min(total//barsize, 100), 1)
    if current == 1: print("\nSTART!", flush=True)
    if current == 0 or current % print_frequency == 0:
        print('[', chr(9608)*progress, ' ', completed, '.'*(barsize-progress), '] ', str(current)+'/'+str(total)+' epochs', sep='', end='\r', flush=True)
    if current == total:
        print("\nFinished", flush=True)


if __name__=="__main__":
    file_path = "dataFile/for_presentation20240202/testing_datacollection.csv"
    df = pd.read_csv(file_path)
    y = df["followingAcceleration"].to_numpy()
    X = df[["distance","precedingAcceleration","precedingSpeed","followingSpeed"]].to_numpy()
    obj = Multiple_LinReg(learning_rate= 0.00001,epochs=100000)
    obj.fit(X,y)

    # for checking efficiency
    reg = LinearRegression().fit(X, y)
    print(reg.coef_)
    print(mean_squared_error(y,obj.predict(X)))
    print(mean_squared_error(y,reg.predict(X)))





