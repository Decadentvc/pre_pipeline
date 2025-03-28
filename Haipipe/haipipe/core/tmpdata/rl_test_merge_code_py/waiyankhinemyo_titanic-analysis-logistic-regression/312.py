import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
#sns.set()
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import date
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = '../input/titanicdataset-traincsv/train.csv'
dataset = pd.read_csv(data)
dataset.shape
dataset.dtypes
dataset.describe()
dataset.head(10)
print("Total number of passengers in the dataset: " + str(len(dataset.index)))
#sns.countplot(x="Survived", data=dataset)
#sns.countplot(x="Survived", hue="Sex", data=dataset)
#sns.countplot(x="Survived", hue="Pclass", data=dataset)
dataset["Age"].plot.hist()
#sns.boxplot(x="Survived", y="Age", data=dataset)
dataset["Pclass"].plot.hist()
#sns.boxplot(x="Pclass", y="Age", data=dataset)
dataset["Fare"].plot.hist(figsize=(10,10))
dataset.info()
#sns.countplot(x="SibSp", data=dataset)
#sns.countplot(x="Parch", data=dataset)
dataset.isnull()
dataset.isnull().sum()
#sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
dataset.head()
dataset.drop("Cabin", axis=1, inplace=True)
dataset.head()
#sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
dataset.dropna(inplace=True)
#sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
dataset.isnull().sum()
dataset.shape
dataset.head()
dataset.Pclass.unique()
dataset.Sex.unique()
dataset.Embarked.unique()
Pcl=pd.get_dummies(dataset["Pclass"],drop_first=True)
Pcl.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = dataset
a = dataset['Sex']
X['Sex'] = le.fit_transform(X['Sex'])
a = le.transform(a)
dataset = X
embark=pd.get_dummies(dataset["Embarked"])
embark.head()

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        
dataset = pd.DataFrame(dataset).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(include_bias=False)
add_engine.fit(dataset)
train_data_x = add_engine.transform(dataset)
train_data_x = pd.DataFrame(train_data_x)
dataset = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        
dataset=pd.concat([dataset,embark,Pcl],axis=1)
dataset.head()
dataset.drop(['PassengerId','Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
dataset.head()
corr_mat=dataset.corr()
#plt.figure(figsize=(13,5))
#sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
#plt.show()
X = dataset.drop("Survived", axis=1)
y = dataset["Survived"]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
#logmodel.fit(X_train,y_train)
#predictions = logmodel.predict(X_test)
#print(predictions)
from sklearn.metrics import classification_report
#print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score 
#accuracy_score(y_test,predictions) 




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
#print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/merge_max_result_rl/waiyankhinemyo_titanic-analysis-logistic-regression/312.npy", { "accuracy_score": score })

