import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
cust_churn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
cust_churn.head()
c_5 = cust_churn.iloc[:,4]
c_5.head()
c_15 = cust_churn.iloc[:,14]
c_15.head()
senior_male_electronic = cust_churn[(cust_churn['gender'] == 'Male') & (cust_churn['SeniorCitizen'] == 1) & (cust_churn['PaymentMethod'] == 'Electronic check')]
senior_male_electronic.head()
customer_total_tenure = cust_churn[(cust_churn['tenure']>70) | (cust_churn['MonthlyCharges']> 100)]
customer_total_tenure.head()
two_mail_yes = cust_churn[(cust_churn['Contract']=='Two year')&(cust_churn['PaymentMethod']=='Mailed check')&(cust_churn['Churn']=='Yes')]
two_mail_yes.head()
custumer_333 = cust_churn.sample(n=333)
custumer_333.head()
cust_churn['Churn'].value_counts()
cust_churn['Contract'].value_counts()
#plt.bar(cust_churn['InternetService'].value_counts().keys().tolist(),cust_churn['InternetService'].value_counts().tolist(), color = 'red')
#plt.xlabel('Categories of Internet Service')
#plt.ylabel('Count')
#plt.title('Distribution of Internet Service')
#plt.hist(cust_churn['tenure'], bins = 30 ,color = 'green')
#plt.title('Distribution of tenure')
#plt.scatter(cust_churn['tenure'], cust_churn['MonthlyCharges'])
#plt.xlabel('Tenure')
#plt.ylabel('Monthly Charges')
#plt.title('MonthlyCharges vs tenure')
#cust_churn.boxplot(column=['tenure'], by=['Contract'])
#plt.xlabel('Contract')
#plt.ylabel('Tenure')
#plt.title('Contract vs Tenure')
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
y = cust_churn[['MonthlyCharges']]
x = cust_churn[['tenure']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
regressor = LinearRegression()
#regressor.fit(x_train, y_train)
#y_predict = regressor.predict(x_test)
from sklearn.metrics import mean_squared_error
#np.sqrt(mean_squared_error(y_test,y_predict))
#print(y_predict[:5]) 
print(y_test[:5]) 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x = cust_churn[['MonthlyCharges']]
y = cust_churn[['Churn']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
regressor = LogisticRegression()
#regressor.fit(x_train,y_train)
#y_predict = regressor.predict(x_test)
#y_predict[:5]
from sklearn.metrics import confusion_matrix, accuracy_score
#confusion_matrix(y_test, y_predict)
#accuracy_score(y_test, y_predict)
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x = cust_churn[['MonthlyCharges','tenure']]
y = cust_churn[['Churn']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
regressor = LogisticRegression()
#regressor.fit(x_train,y_train)
#y_predict = regressor.predict(x_test)
#y_predict[:5]
from sklearn.metrics import confusion_matrix, accuracy_score
#confusion_matrix(y_test,y_predict)
#accuracy_score(y_test, y_predict)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x = cust_churn[['tenure']]
y = cust_churn[['Churn']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=1-0.8, random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
DTree = DecisionTreeClassifier()
#DTree.fit(x_train,y_train)
#y_predict = DTree.predict(x_test)
#y_predict[:5]
from sklearn.metrics import confusion_matrix, accuracy_score
#confusion_matrix(y_test,y_predict)
#accuracy_score(y_test,y_predict)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#rf.fit(x_train,y_train)
#y_predict = rf.predict(x_test)
#y_predict[:5]
#confusion_matrix(y_test,y_predict)
#accuracy_score(y_test,y_predict)




import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#print("start running model training........")
model = RandomForestClassifier(random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
import numpy as np
np.save("haipipe/core/tmpdata/prenotebook_res/himanshumriu_predictive-analysis-basics.npy", { "accuracy_score": score })
