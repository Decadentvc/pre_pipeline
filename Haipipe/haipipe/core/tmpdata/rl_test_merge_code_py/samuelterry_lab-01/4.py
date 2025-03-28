import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/wisconsin-breast-cancer-cytology-features/wisconsin_breast_cancer.csv')
data.head()
new_data = data.dropna()

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        
add_scaler = MinMaxScaler()
new_data = pd.DataFrame(new_data).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(new_data)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
new_data = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        
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
np.save("haipipe/core/tmpdata/merge_max_result_rl/samuelterry_lab-01/4.npy", { "accuracy_score": score })

