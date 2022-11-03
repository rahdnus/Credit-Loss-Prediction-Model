import pandas as pd #To work with dataset
import numpy as np #Math library

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

df_credit = pd.read_csv("data/German_credit_data.csv",index_col=0)
df_credit.head()

df_credit.isnull().sum()/len(df_credit)

print(df_credit.nunique())

df_credit.fillna('Missing', inplace=True)
df_credit.head()
df_credit['Job'] = df_credit['Job'].astype('category')
df_credit_dummy = pd.get_dummies(df_credit, drop_first = True)
df_credit_dummy.head()
X = df_credit_dummy.drop('Risk_good',axis=1)
y = df_credit_dummy['Risk_good']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
# y_test.to_csv("y_test.csv")
log = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
log.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth = 50)
dt.fit(X_train, y_train)
y_pred_log = log.predict(X_test)
y_pred_dt = dt.predict(X_test)
param_grid = { "max_depth" : [3,5,7,9,11,13,15,17,19,21],
             "max_features" : [2,4,6,8,10]}
model = RandomForestClassifier(random_state = 42)
grid_search = GridSearchCV(model, param_grid = param_grid, cv = 5 )
grid_search.fit(X_train, y_train)
rf = RandomForestClassifier(max_depth= 7, max_features= 2)
rf.fit(X_train, y_train)
X_test.to_csv("test.csv")
y_pred_rf = rf.predict(X_test)
pickle.dump(dt,open('models/dtmodel.pkl','wb'))
