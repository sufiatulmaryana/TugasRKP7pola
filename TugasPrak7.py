import pandas as pd
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

from cart import Y

#Melakukan pembacaaan dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names) 
print(pima)

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree'] 
x = pima[feature_cols] # Features
y = pima.label # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)


# Membuat objek DT
# Dapat dioptimalkan dengan menghitung Entropy 
clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Melakukan Pelatihan DT
clf = clf.fit(X_train,y_train)


# Memprediksi
y_pred = clf.predict(X_test)


# Menghitung akurasi model 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#untuk text cart
test_represent = tree.export_text(clf);
print(test_represent)

#untuk plot tree
fgr = plt.figure(figsize=(2,1))
_ = tree.plot_tree(clf,feature_names=x,class_names=y,filled=True)
plt.show()