import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
df = pd.read_csv(r"../input/baseball.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imputer = imputer.fit(df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])
df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']] = imputer.transform(df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])

from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != 'object']
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x
        
add_scaler = RobustScaler()
df = pd.DataFrame(df).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(df)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
df = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        
df.League.replace(['NL', 'AL'], [1, 0], inplace=True)
df = df[df.columns.difference(['RankPlayoffs', 'Team'])]
y = df[['Playoffs']]
y = np.ravel(y)
X = df[df.columns.difference(['Playoffs'])]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
#svc = SVC(kernel='linear').fit(x_train, y_train)
#predictions = svc.predict(x_test)
#print(predictions)
#print(accuracy_score(y_test, predictions))
#print(classification_report(y_test, predictions))




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#print("start running model training........")
model = SVC(random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/merge_max_result_rl/reacherai_my-moneyball/63.npy", { "accuracy_score": score })

