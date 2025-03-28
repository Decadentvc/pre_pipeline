import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/voicegender/voice.csv')
df.columns
df.shape
df.dtypes
colors = ['pink','Lightblue']
data_y = df[df.columns[-1]]
#plt.pie(data_y.value_counts(),colors=colors,labels=['female','male'])
#plt.axis('equal')
#df.boxplot(column = 'meanfreq',by='label',grid=False)
correlation =df.corr()
#sns.heatmap(correlation)
#plt.show()
from sklearn.model_selection import train_test_split
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier()
#rand_forest.fit(Xtrain, ytrain)
#y_pred = rand_forest.predict(Xtest)
from sklearn import metrics, neighbors
from sklearn.metrics import accuracy_score
#print(metrics.accuracy_score(ytest, y_pred))
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(ytest, y_pred))
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
CVFirst = GaussianNB()
#CVFirst = CVFirst.fit(Xtrain, ytrain)
#test_result = cross_val_score(CVFirst, X, y, cv=10, scoring='accuracy')
#print('Accuracy obtained from 10-fold cross validation is:',test_result.mean())
male_funFreq_outlier_index = df[((df['meanfun'] < 0.085) | (df['meanfun'] > 0.180)) &                               (df['label'] == 'male')].index
female_funFreq_outlier_index = df[((df['meanfun'] < 0.165)  | (df['meanfun'] > 0.255)) &                                 (df['label'] == 'female')].index
index_to_remove = list(male_funFreq_outlier_index) + list(female_funFreq_outlier_index)
len(index_to_remove)
data_x = df[df.columns[0:20]].copy()
data2 = data_x.drop(['kurt','centroid','dfrange'],axis=1).copy()
data2.head(3)

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
        
data2 = pd.DataFrame(data2).reset_index(drop=True).infer_objects()
add_engine = PolynomialFeatures(interaction_only=True, include_bias=False)
add_engine.fit(data2)
train_data_x = add_engine.transform(data2)
train_data_x = pd.DataFrame(train_data_x)
data2 = train_data_x.loc[:, ~train_data_x.columns.duplicated()]
        
data2 = data2.drop(index_to_remove,axis=0)
data_y = pd.Series(y).drop(index_to_remove,axis=0)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data2, data_y, train_size=0.8, test_size=1-0.8, random_state=0)
clf1 = RandomForestClassifier()
#clf1.fit(Xtrain, ytrain)
#y_pred = clf1.predict(Xtest)
#print(metrics.accuracy_score(ytest, y_pred))
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier()
#clf2.fit(Xtrain, ytrain)
#y_predict = clf2.predict(Xtest)
#print(metrics.accuracy_score(ytest, y_predict))
clf3 = GaussianNB()
#clf3 = clf3.fit(Xtrain, ytrain)
#y_predd = clf3.predict(Xtest)
#print(metrics.accuracy_score(ytest,y_predd))
from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression()
#clf4.fit(Xtrain,ytrain)
#y_predict4 = clf4.predict(Xtest)
#test_result = cross_val_score(clf3, data2, data_y, cv=10, scoring='accuracy')
#test_result = cross_val_score(clf2, data2, data_y, cv=10,scoring = 'accuracy')
import pylab as pl
labels = ['female', 'male']
#cm = confusion_matrix(ytest,y_pred,labels)  
#print(cm)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax =ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
#pl.show()
from sklearn.metrics import classification_report
#print(classification_report(ytest, y_pred))
#sns.FacetGrid(df, hue='label',size=5).map(sns.kdeplot,"meanfun").add_legend()
#plt.show()
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")
data_x = np.array(df[['meanfreq','meanfun']])
kmeans = KMeans(n_clusters= 2)
kmeans.fit(data_x)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ["g.","b."]  
#for i in range(len(data_x)):
#    plt.plot(data_x[i][0], data_x[i][1], colors[labels[i]], markersize = 10)
    
#plt.scatter(centroids[:,0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
#plt.ylabel('meanfun')
#plt.xlabel('meanfun')
#plt.show()




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
#print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
score = accuracy_score(ytest, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/merge_max_result_rl/datascientist25_gender-recognition-by-voice-using-machine-learning/35.npy", { "accuracy_score": score })

