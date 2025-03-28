import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('data/dataset/hb20007_gender-classification'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataframe = pd.read_csv("data/dataset/hb20007_gender-classification/gender-classification/Transformed Data Set - Sheet1.csv")
dataframe.info()
dataframe.rename(columns={'Favorite Color' :'FavoriteColor', 'Favorite Music Genre':'FavoriteMusicGenre',                           'Favorite Beverage':'FavoriteBeverage', 'Favorite Soft Drink':'FavoriteSoftDrink'}, inplace=True)
from sklearn.preprocessing import LabelEncoder
dataframe=dataframe.apply(LabelEncoder().fit_transform)
dataframe.info()
X = dataframe.drop(['Gender'], axis = 1)
y = dataframe.Gender
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=1-0.8, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
#print("Result for the test set is : {}".format(y_pred))
#print("Score on training set is : {:.2f}%".format(knn.score(X_train, y_train)*100))
#print("Score on test set is : {:.2f}%".format(knn.score(X_test, y_test)*100))
from sklearn.metrics import recall_score, precision_score, confusion_matrix
#print("Recall score", recall_score(y_test, y_pred, average='macro'))
#print("Precision score", precision_score(y_test, y_pred, average='macro'))
#print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred))




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#print("start running model training........")
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/grandmaster619_gender-classification-using-knn-classifier.npy", { "accuracy_score": score })
