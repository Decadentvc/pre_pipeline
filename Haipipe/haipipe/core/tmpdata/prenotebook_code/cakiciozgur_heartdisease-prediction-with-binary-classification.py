import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
print(os.listdir("../input"))
data=pd.read_csv("../input/heart.csv")
data.head()
data.info()
data.isnull().sum()
y=data.target   
x_data =data.drop(["target"],axis=1)  
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
#lr.fit(x_train,y_train)
#print("Accuracy Score:",(lr.score(x_test,y_test))) 




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
#print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/cakiciozgur_heartdisease-prediction-with-binary-classification.npy", { "accuracy_score": score })
