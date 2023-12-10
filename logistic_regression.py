import numpy as np
import warnings
import pandas as pd

data = np.loadtxt("for_presentation20231123/testdata1.csv", delimiter="," , skiprows = 1 ,usecols=(0,2,3))
target = np.loadtxt("for_presentation20231123/testdata1.csv", delimiter="," , skiprows = 1 ,usecols=(1),dtype='str')
# print(data[:5],data.shape)
# print(target[:5],target.shape)

u = np.unique(target)
target_value = np.zeros(target.shape[0])
class_ = len(u)
count = 0
for i in target[:]:
    sorting = np.searchsorted(u,i)
    if (sorting == 2): sorting = 1
    target_value[count] = sorting
    count += 1
            
# print(u)
# print(target.reshape(len(target),1))
#shape[0] -> no. of dataset, shape[1] -> no. of feature

def stardardize(X):
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
    return X

def sigmoid(z):
    sig = 1/(1+np.e**(-z))
    return sig

# ref: f1 score https://www.v7labs.com/blog/f1-score-guide
# y = actual, y_hat = prediction
def f1_score(y,y_hat):
    if type(y) != list:
        y = y.tolist()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            # correctly predict as positive
            TP += 1
        elif y[i] == 1 and y_hat[i] == 0:
            # wrongly predict as negetive
            FN += 1
        elif y[i] == 0 and y_hat[i] == 1:
            # wrongly predict as positive
            FP += 1
        elif y[i] == 0 and y_hat[i] == 0:
            #correctly predicted as negative
            TN += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score_ = 2*precision*recall/(precision+recall)
    return f1_score_
    
# cannot use now, dont know why
def cost(X,y,theta):
    warnings.filterwarnings('ignore')
    z = np.dot(X,theta)
    cost0 = np.dot(y.T,np.log(sigmoid(z)))
    cost1 = np.dot((1-y).T,np.log(1-sigmoid(z)))
    cost_ = -1 * (cost0+cost1)/(len(y)) 
    return cost_
    
def cost_prime(X,y,theta):
    z = np.dot(X,theta)
    cost_prime_ = np.dot(X.T,(sigmoid(z)-y.reshape(len(y),1)))
    return cost_prime_


class LogReg(): 
    def __init__(self,learning_rate = 0.01,epochs =100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self,X,y):
        warnings.filterwarnings('ignore')
        X = np.c_[(np.ones((X.shape[0],1)),X)]
        cost_list = np.zeros(self.epochs,)
        total = self.epochs
        if self.weights == None:
            np.random.seed(1)
            self.weights = np.random.rand(X.shape[1],1)

        for i in range(self.epochs):
            self.weights = self.weights - self.learning_rate * cost_prime(X,y,self.weights)
            cost_list[i] = cost(X,y,self.weights)
            print_progessbar(total,i+1)
        warnings.filterwarnings('default')
        return cost_list
    
    def predict(self,X):
        z = np.dot((np.c_[np.ones((X.shape[0],1)),X]),self.weights)
        weight_list = self.weights.tolist()
        result = pd.DataFrame(weight_list,columns=["Coefficient"],index=['Constant','distance','speed','followingspeed'])
        print(result)
        lis = []
        for i in sigmoid(z):
            if i>0.5 and i<=1:
                lis.append(1)
            elif i>=0 and i<=1:
                lis.append(0)
            else:
                print("Error: There is unclassified result.")
                return []   
        return lis

def print_progessbar(total,current,barsize = 60):
    progress = int(current*barsize/total)   # stardardize current into bar
    completed = str(int(current/total*100))+"%"      # showing progress how many percent
    print_frequency = max(min(total//barsize, 100), 1)
    if current == 1: print("\nSTART!", flush=True)
    if current == 0 or current % print_frequency == 0:
        print('[', chr(9608)*progress, ' ', completed, '.'*(barsize-progress), '] ', str(current)+'/'+str(total)+' epochs', sep='', end='\r', flush=True)
    if current == total:
        print("\nFinished", flush=True)

LogRegRegressor = LogReg(learning_rate= 0.0001,epochs = 100)
LogRegRegressor.fit(data,target_value)

predicted = LogRegRegressor.predict(data)

print(f"f_1 score is : {f1_score(target_value,predicted)}")