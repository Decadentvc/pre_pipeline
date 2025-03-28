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

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        
df = pd.DataFrame(df).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(interaction_only=True, include_bias=False)
add_engine.fit(df)
train_data_x = add_engine.transform(df)
train_data_x = pd.DataFrame(train_data_x)
df = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        
df.League.replace(['NL', 'AL'], [1, 0], inplace=True)
df = df[df.columns.difference(['RankPlayoffs', 'Team'])]
y = df[['Playoffs']]
y = np.ravel(y)
X = df[df.columns.difference(['Playoffs'])]
from sklearn.model_selection import train_test_split
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
#model.fit(x_train, y_train)
#y_pred = model.predict(x_validation_varible)
#score = accuracy_score(y_validation_varible, y_pred)
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(model, x_train, y_train,cv=4)
import numpy as np
#np.save("haipipe/core/tmpdata/merge_max_result_rl/reacherai_my-moneyball/15.npy", { "accuracy_score": score })
np.save("haipipe/core/tmpdata/rl_cross_val_res/reacherai_my-moneyball/15.npy", { "accuracy_score": cross_score })


