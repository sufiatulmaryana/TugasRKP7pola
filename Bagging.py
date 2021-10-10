#Impor Library
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import stats
from sklearn import datasets
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import _fit_one, make_pipeline
from sklearn.ensemble import BaggingClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.utils.fixes import loguniform
from sklearn import metrics

from cart import Y_train

#Load cancer dataset
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

#membagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))

#tanpa bagging
model = pipeline.fit(X_train,y_train)
model.score(X_test,y_test)
print('Model Linear test Score: %.3f, ' %model.score(X_test, y_test), 'Model Linear training Score: %.3f' %model.score(X_train, y_train))

#Dengan Bagging
bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
max_features=8, max_samples=80, random_state=1, n_jobs=5)
bgclassifier.fit(X_train, y_train)
print('Model Bagging test Score: %.3f, ' %bgclassifier.score(X_test, y_test), 'Model Bagging training Score: %.3f' %bgclassifier.score(X_train, y_train))



#plt.figure
#plt.xlabel("number of feature")
#plt.ylabel("score")
#plt.plot(range(1,len(rfe.grid_scores_+1)),rfe.grid_scores_)
#plt.show()