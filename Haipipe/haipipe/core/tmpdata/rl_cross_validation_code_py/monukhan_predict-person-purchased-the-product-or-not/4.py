import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
dataset.head(n= 10)  
dataset.isnull().sum()
dataset.info()
dataset.describe()
#sns.countplot(dataset['Purchased'])
#plt.title('Distribution of Purchased or not')
#plt.xlabel('Purchased or not')
#plt.ylabel('Frequency')
#plt.show()
#plt.figure(figsize = (10,6))
#plt.hist(dataset['Age'], bins  = 6, color = 'blue', rwidth = 0.98)
#plt.title('Distribution of Age')
#plt.xlabel('Different Ages')
#plt.ylabel('Frequency')
#plt.figure(figsize = (10,6))
#plt.hist(dataset['EstimatedSalary'], bins = 10, color = 'green',rwidth = 0.97)
#plt.title('Distribution of EstimatedSalaries')
#plt.xlabel('Different Salaries')
#plt.ylabel('Frequency')
#sns.pairplot(dataset, hue = 'Purchased')
#sns.heatmap(dataset.corr(), annot = True, cmap = "RdYlGn")
X = dataset.iloc[:,[2,3]].values
print(X)
y = dataset.iloc[:,4].values
print(y)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)

from sklearn.decomposition import IncrementalPCA
import pandas as pd
import numpy as np
        
X = pd.DataFrame(X).reset_index(drop=True).infer_objects()
add_engine = IncrementalPCA()
cols = list(X.columns)
add_engine.fit(X)

train_data_x = add_engine.transform(X)
X = pd.DataFrame(train_data_x, columns=cols[:train_data_x.shape[1]])
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
print(X_train)
print(X_test)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
#    knn.fit(X_train, y_train)
#    score = knn.predict(X_test)
#    error_rate.append(1-score.mean())
#plt.figure(figsize =(10,6))
#plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
#plt.title('Error rate vs K-value')
#plt.xlabel('K')
#plt.ylabel('Error Rate')
#plt.show()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
#cm1 = confusion_matrix(y_test,y_pred)
#print(cm1)   
#ac1 = accuracy_score(y_test, y_pred)*100
#print(ac1)
#plt.figure(figsize = (12,8))
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],                color = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('kNN (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#plt.figure(figsize = (12,8))
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],                color = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('kNN (Testing set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#print("start running model training........")
model = KNeighborsClassifier()
#model.fit(X_train, y_train)
#y_pred = model.predict(x_validation_varible)
#score = accuracy_score(y_validation_varible, y_pred)
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(model, X_train, y_train,cv=4)
import numpy as np
#np.save("haipipe/core/tmpdata/merge_max_result_rl/monukhan_predict-person-purchased-the-product-or-not/4.npy", { "accuracy_score": score })
np.save("haipipe/core/tmpdata/rl_cross_val_res/monukhan_predict-person-purchased-the-product-or-not/4.npy", { "accuracy_score": cross_score })


