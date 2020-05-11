import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


#read data
data = pd.read_csv("heart.csv")
print(data.head())


sns.countplot(x="target", data=data, palette = "bwr")
plt.show()

sns.countplot(x='sex', data=data,palette="mako_r")
plt.xlabel("Sex (0=female, 1=male)")
plt.show()

plt.scatter(x=data.age[data.target==1],y=data.thalach[(data.target==1)],c="green")
plt.scatter(x=data.age[data.target==0],y=data.thalach[(data.target==0)],c="black")
plt.legend(["Disease","Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum heart rate")
plt.show()

X= data.iloc[:,:-1].values
Y= data.iloc[:,13].values
print(data.head())

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)


#Showing accuracy,precision and recall
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))



#Finding the value of K
error_rate=[]
for i in range(1,40):
   
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))
#plotting the error_rate vs k value

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')




