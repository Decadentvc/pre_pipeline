import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/wisconsin-breast-cancer-cytology-features/wisconsin_breast_cancer.csv')
data.head()
new_data = data.dropna()
new_data.info()
import seaborn as sns
#sns.pairplot(data=new_data, hue="class", palette="Set2" ,diag_kind="hist")
X = new_data.iloc[0:683 , 1:10]
print(X)
y = new_data.iloc[0:683 , 10:11]
print(y)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.svm import SVC
model = SVC()
#model.fit(X_train, y_train)
#pred = model.predict(X_test)
#print(pred[0:10])
print(y_test[0:10])
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test,pred))
from sklearn.metrics import classification_report
#print(classification_report(y_test,pred))




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#print("start running model training........")
model = SVC(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/samuelterry_lab-01.npy", { "accuracy_score": score })
