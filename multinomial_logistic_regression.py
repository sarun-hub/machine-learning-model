import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# data = np.loadtxt("for_presentation20231123/testdata1.csv", delimiter="," , skiprows = 1 ,usecols=(0,2,3))
# target = np.loadtxt("for_presentation20231123/testdata1.csv", delimiter="," , skiprows = 1 ,usecols=(1),dtype='str')
# print(data[:5],data.shape)
# print(target[:5],target.shape)

# target is the list of "Acceleration", "Brake", "No Action"
def get_unique_value(target):
    u = np.unique(target)
    target_value = []
    # class_ = len(u)
    for i in target[:]:
        sorting = np.searchsorted(u,i)
        target_value.append(sorting)
    # taget_value is conversion of target to number A=0, B=1,N=2
    return target_value

    
class Multinomial_Logistic_Regression():
    def __init__(self,learning_rate = 0.01, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    # Y is list of target_value (might be changed to target later)
    def one_hot_encode(self,Y):
        encoded_list = []
        class_ = int(max(Y))
        for element in Y:
            encoded_ = np.zeros(class_+1)
            encoded_[element] = 1
            encoded_list.append(encoded_)
        encoded_list = np.array(encoded_list)
        return encoded_list
    
    #ref: https://en.wikipedia.org/wiki/Softmax_function
    def soft_max(self,z):
        soft = np.exp(z).T/(np.sum(np.exp(z),axis = 1)).T
        return soft
    
    # T = one_hot_encoding (y-actual)
    # O = softmax value (y-predict)
    # X = input
    def cost_prime(self,X,Ti,Oi):
        n = X.shape[0]
        cost_prime_ = -np.dot(X.T,(Ti-Oi.T))/n
        return cost_prime_
    
    def cost(self,X,Ti,Oi):
        n = X.shape[0]
        cost_ = -np.sum(np.dot(Ti,np.log(Oi)))/n
        return cost_
        
    def show_error_graph(self,epochs_list,error_list):
        plt.figure()
        plt.plot(epochs_list,error_list)
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.title("Error minimization")
        plt.show()
    
    def fit(self,X,y):
        n = X.shape[0]
        epochs_list = []
        error_list = []
        if self.weights == None:
            np.random.seed(1)
            self.weights = np.random.rand(X.shape[1],3)
        for i in range(self.epochs):
            Ti = self.one_hot_encode(y)
            z = np.dot(X,self.weights)
            Oi = self.soft_max(z)
            Debug =False
            if Debug == True:
                print(Ti.shape)
                print(z.shape)
                print(Oi.shape)
                return 
            cost_prime_ = self.cost_prime(X,Ti,Oi)
            error = self.cost(X,Ti,Oi)
            error_list.append(error)
            epochs_list.append(i)
            self.weights = self.weights - self.learning_rate*cost_prime_
            print_progessbar(self.epochs,i+1)
        self.show_error_graph(epochs_list,error_list)
        


    def predict(self,X):
        z = np.dot(X,self.weights)
        weight_list = self.weights.T.tolist()
        # result = pd.DataFrame(weight_list,columns=["Class 1","Class 2","Class 3"],index=['Constant','distance','speed','followingspeed'])
        # print(result)
        Oi = self.soft_max(z)
        lis = []
        count = 0
        for element in Oi.T.tolist():
            max_position = element.index(max(element))
            lis.append(max_position)
            count += 1
            # print(element)
            if count <0:
                return []
            if max_position > X.shape[1] and max_position <0:
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


