import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,mean_absolute_error

class Airfoil:
    
    parameters=[]
    def __init__(self):
        pass
        
    def train(self,path):
        df=pd.read_csv(path,header=None)
        df=np.array(df)
        n1,c1=df.shape
        c1-=1
        actual_labels=df[:,c1:]
        li=[1.0]*n1
        df=df[:,:c1]
        scaler = StandardScaler()
        df=scaler.fit_transform(df)
        df=pd.DataFrame(df)
        df.insert(0,"const",li,True)
        n,c=df.shape
        df=np.array(df)
        actual_labels=np.array(actual_labels)
        self.find_parameters(df,actual_labels,n,c)
    
    def find_parameters(self,X_train,y_train,n,c):
      self.parameters = np.random.rand(c)
      self.parameters=np.expand_dims(self.parameters, axis=1)
      print(self.parameters)
      alpha=0.4
      for j in range(0,100):
        predlist=X_train.dot(self.parameters)
        loss=predlist-y_train
        gradient=X_train.T.dot(loss)/n
        self.parameters=self.parameters-(alpha*gradient)
      print(self.parameters)


        
    def predict(self,path):
        df=pd.read_csv(path)
        df=np.array(df)
        n,c=df.shape
        print(n)
        print(c)
        li=[1.0]*n
        c=c-1
        df=df[:,:c]
        scaler = StandardScaler()
        df=scaler.fit_transform(df)
        df=pd.DataFrame(df)
        df.insert(0,"const",li,True)
        df=np.array(df)
        predlist=df.dot(self.parameters)
        return predlist


model3 = Airfoil()
model3.train('/content/drive/My Drive/airfoil.csv') # Path to the train.csv will be provided
prediction3 = model3.predict('/content/drive/My Drive/test_data/airfoil_test.csv') # Path to the test.csv will be provided
df1=pd.read_csv('/content/drive/My Drive/test_data/airfoil_test.csv')
print(df1.shape)
df1=np.array(df1)
labels=df1[:,5:]
# print(labels)
mae = mean_absolute_error(labels, prediction3)
mse = mean_squared_error(labels, prediction3)
print(mae)
print(mse)
r2_score(labels,prediction3)


