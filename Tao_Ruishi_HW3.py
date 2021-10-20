#Read the dataset into a dataframe. (1)
import pandas as pd
pp=pd.read_csv("Titanic (1).csv")
LL=pd.DataFrame(pp)
print(LL.head())
#Explore the dataset and determine what is the target variable. (1)
print(LL.head())
print(LL.shape)
# the target variable will be "Survived"
#Drop factor(s) that are not likely to be relevant for logistic regression. (2)
LL.drop(LL.columns[0],axis=1,inplace=True)
print(LL)
#Make sure there are no missing values. (1)
print(LL.isnull().sum())
#there are no missing values
#Plot count plots of each of the remaining factors (including the target variable). (2)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
ac=sns.countplot(x='Class',data=pp)
plt.show()
aq=sns.countplot(x='Sex',data=pp)
plt.show()
aa=sns.countplot(x='Age',data=pp)
plt.show()
ax=sns.countplot(x='Survived',data=pp)
plt.show()
#Convert all categorical feature variables into dummy variables. (2)
df1=pd.get_dummies(LL,
                   columns=["Class","Sex","Age","Survived"])
print(df1)
df1.drop(["Class_Crew","Sex_Male","Age_Child","Survived_No"],axis=1,inplace=True)
print(df1)
X=df1.iloc[:,:5]
y=df1.iloc[:,-1]
#Partition the data into train and test sets (70/30). Use random_state = 2021. (1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2021,stratify=y)
#Fit the training data to a logistic regression model. (1)
from sklearn.linear_model import LogisticRegression
logReg=LogisticRegression(random_state=2021)
logReg.fit(X_train,y_train)
y_pred=logReg.predict(X_test)
#Display the accuracy of your predictions for survivability. (2)
from sklearn.metrics import accuracy_score
predictions=logReg.predict(X_test)
print(accuracy_score(predictions,y_test))

#Plot the lift curve. Hint: https://scikit-plot.readthedocs.io/en/stable    /metrics.html  (2)
import scikitplot as skplt
y_p=logReg.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test,y_p)
plt.show()
#Plot the confusion matrix along with the labels (Yes, No).  (2)
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
y_pred=logReg.predict(X_test)
plot_confusion_matrix(logReg,X_test,y_test)
plt.show()
#Now, display the predicted value of the survivability of a male adult passenger traveling in 3rd class. (3)
male=pd.DataFrame({"Class_1st":[0],"CLass_2nd":[0],"Class_3rd":[1],"Sex_Female":[0],"Age_Adult":[1]})
outcome=logReg.predict(male)
print(outcome)
