import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,roc_auc_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
df=pd.read_csv('../input/heart-patients/US_Heart_Patients.csv')
df.describe()
df.isnull().sum()

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        
no_m_v = pd.DataFrame(no_m_v).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(include_bias=False)
add_engine.fit(no_m_v)
train_data_x = add_engine.transform(no_m_v)
train_data_x = pd.DataFrame(train_data_x)
no_m_v = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        
no_m_v=df.dropna(axis=0)
data_w_d=pd.get_dummies(no_m_v,drop_first=True)

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
data_w_d = pd.DataFrame(data_w_d).reset_index(drop=True).infer_objects()
cat_train_x, num_train_x = catch_num(data_w_d)
add_scaler.fit(num_train_x)
num_train_x = pd.DataFrame(add_scaler.transform(num_train_x), columns=list(num_train_x.columns)).reset_index(drop=True).infer_objects()
data_w_d = pd.concat([cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],axis=1)
        
data_w_d.columns
y=data_w_d['TenYearCHD']
x=data_w_d.drop(['TenYearCHD'],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)
models={'Logistic Regression':LogisticRegression(),'KNN':KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
def fit_and_plot(models,x_train,x_test,y_train,y_test):
    model_scores={}
#    for name,model in models.items():
#        model.fit(x_train,y_train)
#        model_scores[name]=model.score(x_test,y_test)
    return model_scores
#scores=fit_and_plot(models,x_train,x_test,y_train,y_test)
#scores
train_scores=[]
test_scores=[]
knn=KNeighborsClassifier()
neighbors=range(1,41)
#for i in neighbors:
#    knn.fit(x_train,y_train)
#    knn.set_params(n_neighbors=i)
#    train_scores.append(knn.score(x_train,y_train))
#    test_scores.append(knn.score(x_test,y_test))
#plt.plot(neighbors,train_scores,label='train',color='orange')
#plt.plot(neighbors,test_scores,label='test',color='darkblue')
#plt.legend()
#plt.show()
#print(f'Test maximum score:{max(test_scores)*100:.2f}%')
log_grid={'C':np.logspace(-4,4,20),'solver':['liblinear']}
random_grid={'n_estimators':[50,100,150],'max_depth':[None,5,10],'max_features':['auto','sqrt'],'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,3]}
rs_log_search=RandomizedSearchCV(LogisticRegression(),param_distributions=log_grid,n_iter=5,cv=5,verbose=True)
rs_random_search=RandomizedSearchCV(RandomForestClassifier(),param_distributions=random_grid,n_iter=5,cv=5,verbose=True)
#rs_log_search.fit(x_train,y_train)
#rs_random_search.fit(x_train,y_train)
#print(rs_log_search.score(x_test,y_test))
#print(rs_random_search.score(x_test,y_test))




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#print("start running model training........")
model = RandomForestClassifier(random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/merge_max_result_rl/kulchorobeknazarov_heart-disease-prediction/50.npy", { "accuracy_score": score })


