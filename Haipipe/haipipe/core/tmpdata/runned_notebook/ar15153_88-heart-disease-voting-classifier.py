import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/heart.csv")
df.head()
df.isna().sum()
print(len(df))
y = df['target']
x = df.loc[:, :'thal']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],    voting = 'soft')
#voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score 
#for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#cm = confusion_matrix(y_test, y_pred)
#plt.figure(figsize=(24,12))
#plt.subplot(2,3,1)
#plt.title("Logistic Regression Confusion Matrix")
#sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#print("start running model training........")
model = SVC(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/ar15153_88-heart-disease-voting-classifier.npy", { "accuracy_score": score })
