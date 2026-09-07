# 0.Preprocess + EDA + feature selection
# 1.Extract input and output columns
# 2.scale the value 
# 3.Train test split 
# 4.train the Model 
# 5.Evaluate the model/model_selection
# 6.Deploy the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pickle
df = pd.read_csv("placement.csv")
df = df.iloc[:,1:]
df.head()
#print(df)


plt.scatter(df['cgpa'],df['iq'],c = df["placement"])
# plt.show()

X = df.iloc[:,0:2]
Y = df.iloc[:,-1]
# print(X)
# print(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=42)
# print("X_train:")
# print(X_train)

# print("X_test:")
# print(X_test)

# print("y_train:")
# print(y_train)

# print("y_test:")
# print(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#print(X_train)

X_test = scaler.transform(X_test)
#print(X_test)

# model training 
clf = LogisticRegression()
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)

plot_decision_regions(X_train,y_train.values,clf=clf,legend= 2)
# plt.show()

pickle.dump({"model": clf, "scaler": scaler}, open("placement_model.pkl", "wb"))

saved_data = pickle.load(open("placement_model.pkl", "rb"))

model = saved_data["model"]
scaler = saved_data["scaler"]

new_student = [[8.34, 120]]  # cgpa, iq
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)
print(prediction)