from os import name
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats.stats import mode
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import random as rnd 
from sklearn import tree

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names) 
boston['MEDV'] = boston_dataset.target
names = boston_dataset.feature_names

array = boston.values 
X = array[:,0:13]
Y = array[:,13] 
print(X) 
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

#model = DecisionTreeRegressor(max_leaf_nodes = 20)
model = DecisionTreeRegressor(criterion='squared_error', max_depth=None, max_features=None, max_leaf_nodes=50,  min_impurity_decrease=0.0, ccp_alpha=0, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best')

#evaluasi
print("evaluasi")
rt = model.fit(X_train, Y_train) 

#untuk text cart
#test_represent = tree.export_text(rt);
#print(test_represent)

#untuk plot tree
fgr = plt.figure(figsize=(2,1))
_ = tree.plot_tree(model,feature_names=names,filled=True)
plt.show()



rnd.seed(123458)
X_new = X[rnd.randrange(X.shape[0])] 
X_new = X_new.reshape(1,13)

#Prediksi Model
YHat = model.predict(X_new)

df = pd.DataFrame(X_new, columns = names) 
df["Predicted Price"] = YHat
df.head(1)

YHat = model.predict(X_test) 
print(YHat)

r2 = r2_score(Y_test, YHat) 
print("R-Squared = ", r2)

