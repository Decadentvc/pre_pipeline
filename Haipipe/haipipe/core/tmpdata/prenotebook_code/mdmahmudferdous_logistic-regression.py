import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/titanic_cleaned.csv")
df.head()
X=df.drop(columns=['Survived'])
y=df[['Survived']]
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
#LR=LogisticRegression().fit(X_train,y_train)
#y_pred=LR.predict(X_test)
from sklearn.metrics import classification_report
#print("Logistic Regression Report: ", classification_report(y_pred, y_test))




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
#print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/mdmahmudferdous_logistic-regression.npy", { "accuracy_score": score })
