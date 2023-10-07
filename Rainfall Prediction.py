#!/usr/bin/env python
# coding: utf-8

# # Rainfall Prediction

# In[1]:


import pandas as pd
import numpy as np 

df = pd.read_csv(r"C:\Users\HP\Downloads\rainfallpred\dataset1.csv")
df


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df.describe()


# In[4]:


df.dtypes


# In[5]:


df["rainfall"].unique()


# In[6]:


df = df.dropna()


# In[7]:


df.isnull().sum()


# In[8]:


Value_Mapping1 = {'no':0, 'yes':1}
df["rainfallpred"] = df['rainfall'].map(Value_Mapping1)
df = df.drop(['rainfall'],axis = 1)
df


# In[9]:


y = df["rainfallpred"]
X = df.drop("rainfallpred", axis = 1)


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(X_train,y_train)
forest.score(X_test,y_test)


# In[103]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\randomforestregressor.pkl","wb") as a:
    pickle.dump(forest,a)

with open(r"C:\Users\HP\Downloads\rainfallpred\randomforestregressor.pkl","rb") as a:
    mod1 = pickle.load(a)


# In[22]:


from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)
regressor.score(X_test,y_test)


# In[104]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\decisiontreeregressor.pkl","wb") as b:
    pickle.dump(regressor,b)

with open(r"C:\Users\HP\Downloads\rainfallpred\decisiontreeregressor.pkl","rb") as b:
    mod2 = pickle.load(b)


# In[23]:


from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(X_train, y_train) 
y_pred = vot_hard.predict(X_test) 
score = accuracy_score(y_test, y_pred) 
print("Hard Voting Score % d" % score) 
vot_soft = VotingClassifier(estimators = estimator, voting ='soft') 
vot_soft.fit(X_train, y_train) 
y_pred = vot_soft.predict(X_test) 
score = accuracy_score(y_test, y_pred) 
print("Soft Voting Score % d" % score) 


# In[106]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\votingclassifiersoft.pkl","wb") as c:
    pickle.dump(vot_soft,c)

with open(r"C:\Users\HP\Downloads\rainfallpred\votingclassifiersoft.pkl","rb") as c:
    mod3 = pickle.load(c)
    
import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\votingclassifierhard.pkl","wb") as d:
    pickle.dump(vot_hard,d)

with open(r"C:\Users\HP\Downloads\rainfallpred\votingclassifierhard.pkl","rb") as d:
    mod4 = pickle.load(d)


# In[24]:


import xgboost as xgb 
my_model = xgb.XGBClassifier() 
my_model.fit(X_train, y_train) 
y_pred = my_model.predict(X_test) 
my_model.score(X_test,y_test)


# In[107]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\clfgbc.pkl","wb") as e:
    pickle.dump(my_model,e)

with open(r"C:\Users\HP\Downloads\rainfallpred\clfgbc.pkl","rb") as e:
    mod5 = pickle.load(e)


# In[108]:


from sklearn.metrics import mean_squared_error as MSE 
import xgboost as xg 
xgb_r = xg.XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10, seed = 123) 
xgb_r.fit(X_train, y_train) 
xgb_r.score(X_test,y_test)
pred = xgb_r.predict(X_test) 
rmse = np.sqrt(MSE(y_test, pred)) 
print("RMSE : % f" %(rmse)) 


# In[111]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\xgboost.pkl","wb") as f:
    pickle.dump(xgb_r,f)

with open(r"C:\Users\HP\Downloads\rainfallpred\xgboost.pkl","rb") as f:
    mod6 = pickle.load(f)


# In[26]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)  


# In[113]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\gaussiannaivebayes.pkl","wb") as g:
    pickle.dump(gnb,g)

with open(r"C:\Users\HP\Downloads\rainfallpred\gaussiannaivebayes.pkl","rb") as g:
    mod7 = pickle.load(g)


# In[27]:


from sklearn.svm import SVC 
clf = SVC(kernel='linear') 
clf.fit(X_train, y_train) 
clf.score(X_test,y_test)


# In[114]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\supportvectorclassifier.pkl","wb") as h:
    pickle.dump(clf,h)

with open(r"C:\Users\HP\Downloads\rainfallpred\supportvectorclassifier.pkl","rb") as h:
    mod8 = pickle.load(h)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
forestclass=RandomForestClassifier()
forestclass.fit(X_train,y_train)
forestclass.score(X_test,y_test)


# In[115]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\randomforestclassifier.pkl","wb") as i:
    pickle.dump(forestclass,i)

with open(r"C:\Users\HP\Downloads\rainfallpred\randomforestclassifier.pkl","rb") as i:
    mod9 = pickle.load(i)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
decisiontreeclass = DecisionTreeClassifier()  
decisiontreeclass.fit(X_train, y_train)
decisiontreeclass.score(X_test,y_test)


# In[116]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\decisiontreeclassifier.pkl","wb") as j:
    pickle.dump(decisiontreeclass,j)

with open(r"C:\Users\HP\Downloads\rainfallpred\decisiontreeclassifier.pkl","rb") as j:
    mod10 = pickle.load(j)


# In[30]:


from sklearn.svm import SVR
clf1 = SVR(kernel='linear') 
clf1.fit(X_train, y_train) 
clf1.score(X_test,y_test)


# In[117]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\supportvectorregression.pkl","wb") as k:
    pickle.dump(clf1,k)

with open(r"C:\Users\HP\Downloads\rainfallpred\supportvectorregression.pkl","rb") as k:
    mod11 = pickle.load(k)


# In[32]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# In[118]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\linearregression.pkl","wb") as l:
    pickle.dump(lr,l)

with open(r"C:\Users\HP\Downloads\rainfallpred\linearregression.pkl","rb") as l:
    mod12 = pickle.load(l)


# In[33]:


from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
fit=lor.fit(X_train,y_train)
score = lor.score(X_test,y_test)
score


# In[119]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\logisticregression.pkl","wb") as m:
    pickle.dump(lor,m)

with open(r"C:\Users\HP\Downloads\rainfallpred\logisticregression.pkl","rb") as m:
    mod13 = pickle.load(m)


# In[36]:


from sklearn.linear_model import Ridge,Lasso
rd= Ridge(alpha=0.4)
ls= Lasso(alpha=0.3)
rd.fit(X_train,y_train)
ls.fit(X_train,y_train)
rd.score(X_test,y_test)
ls.score(X_test,y_test)


# In[120]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\ridge.pkl","wb") as n:
    pickle.dump(rd,n)

with open(r"C:\Users\HP\Downloads\rainfallpred\ridge.pkl","rb") as n:
    mod14 = pickle.load(n)

import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\lasso.pkl","wb") as o:
    pickle.dump(ls,o)

with open(r"C:\Users\HP\Downloads\rainfallpred\lasso.pkl","rb") as o:
    mod15 = pickle.load(o)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = knn.predict(X_test)
knn.score(X_test,y_test)


# In[121]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\kneighborclassifier.pkl","wb") as p:
    pickle.dump(knn,p)

with open(r"C:\Users\HP\Downloads\rainfallpred\kneighborclassifier.pkl","rb") as p:
    mod16 = pickle.load(p)


# In[42]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[43]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clfm = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clfm.score(X_test,y_test)


# In[123]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\mlpclassifier.pkl","wb") as q:
    pickle.dump(clfm,q)

with open(r"C:\Users\HP\Downloads\rainfallpred\mlpclassifier.pkl","rb") as q:
    mod17 = pickle.load(q)


# In[44]:


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
clfmr = MLPRegressor(random_state=1, max_iter=300).fit(X_train, y_train)
clfmr.score(X_test,y_test)


# In[124]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\mlpregressor.pkl","wb") as r:
    pickle.dump(clfmr,r)

with open(r"C:\Users\HP\Downloads\rainfallpred\mlpregressor.pkl","rb") as r:
    mod18 = pickle.load(r)


# In[48]:


from sklearn.linear_model import Perceptron
clfp = Perceptron(tol=1e-3, random_state=0)
clfp.fit(X_train, y_train)
clfp.score(X_test,y_test)


# In[126]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\perceptron.pkl","wb") as s:
    pickle.dump(clfp,s)

with open(r"C:\Users\HP\Downloads\rainfallpred\perceptron.pkl","rb") as s:
    mod19 = pickle.load(s)


# In[55]:


from sklearn.model_selection import RepeatedStratifiedKFold
models = []
models.append(('lr', LogisticRegression()))
models.append(('knn', KNeighborsClassifier()))
models.append(('tree', DecisionTreeClassifier()))
models.append(('nb', GaussianNB()))
models.append(('svm', SVC(probability=True)))
ensemble = VotingClassifier(estimators=models, voting='soft')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=1)
# return mean score
scores
scores.mean()


# In[60]:


from sklearn import decomposition
pca = decomposition.PCA()
fa = decomposition.FactorAnalysis()
pca.fit_transform(X_train)
pca.transform(X_test)
pca.n_components_


# In[127]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\decomposition.pkl","wb") as t:
    pickle.dump(pca,t)

with open(r"C:\Users\HP\Downloads\rainfallpred\decomposition.pkl","rb") as t:
    mod20 = pickle.load(t)


# In[64]:


from sklearn.ensemble import AdaBoostClassifier
clfadc = AdaBoostClassifier(n_estimators=10,learning_rate=1)
clfadc.fit(X_train,y_train)
clfadc.score(X_test,y_test)


# In[128]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\adaboostclassifier.pkl","wb") as u:
    pickle.dump(clfadc,u)

with open(r"C:\Users\HP\Downloads\rainfallpred\adaboostclassifier.pkl","rb") as u:
    mod21 = pickle.load(u)


# In[66]:


from sklearn.ensemble import AdaBoostRegressor
clfadr = AdaBoostRegressor(n_estimators=20,learning_rate=0.1)
clfadr.fit(X_train,y_train)
clfadr.score(X_test,y_test)


# In[129]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\adaboostregressor.pkl","wb") as v:
    pickle.dump(clfadr,v)

with open(r"C:\Users\HP\Downloads\rainfallpred\adaboostregressor.pkl","rb") as v:
    mod22 = pickle.load(v)


# In[88]:


from sklearn.ensemble import GradientBoostingClassifier
clfgbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1)
clfgbc.fit(X_train, y_train)
clfgbc.score(X_test,y_test)


# In[131]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\gradientboostingclassifier.pkl","wb") as w:
    pickle.dump(clfgbc,w)

with open(r"C:\Users\HP\Downloads\rainfallpred\gradientboostingclassifier.pkl","rb") as w:
    mod23 = pickle.load(w)


# In[68]:


from sklearn.ensemble import GradientBoostingRegressor
clfgbr = GradientBoostingRegressor(n_estimators = 100, learning_rate = 1.0, max_depth = 1)
clfgbr.fit(X_train, y_train)
clfgbr.score(X_test,y_test)


# In[97]:


import pickle

with open(r"C:\Users\HP\Downloads\rainfallpred\clfgbr.pkl","wb") as x:
    pickle.dump(clfgbr,x)

with open(r"C:\Users\HP\Downloads\rainfallpred\clfgbr.pkl","rb") as x:
    mod24 = pickle.load(x)

