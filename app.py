import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

df = pd.read_csv(r"C:\Users\jayas\Downloads\HEIGHT & WEIGHT\SOCR-HeightWeight.csv")

df.head()

#convert the units of height and weight

df['Weight_kg'] = df['Weight(Pounds)']*0.453592
df['Height(feet.Inches)'] = df['Height(Inches)'] // 12 + (df['Height(Inches)'] % 12) / 10
df.columns

df = df.drop(['Index','Height(Inches)','Weight(Pounds)'],axis = 1)
df.head()

df.isna().sum()

df.corr()

#checking the Outliners using the boxplot

sns.boxplot(x = df['Weight_kg'])
sns.boxplot(x = df['Height(feet.Inches)'])

plt.title('Height Vs Weight')
sns.scatterplot(x = df['Weight_kg'],y = df['Height(feet.Inches)'])

x = df.iloc[:,1]
y = df.iloc[:,0]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_sc = sc.fit_transform(x.values.reshape(-1,1))
y_sc = sc.fit_transform(y.values.reshape(-1,1))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size=0.20)

x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
le = LinearRegression()
le.fit(x_train,y_train)
le_pred = le.predict(x_test)
le_acc = r2_score(le_pred,y_test)
le_mse = mean_squared_error(le_pred,y_test)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
dt_pred = dtr.predict(x_test)
dt_acc = r2_score(dt_pred,y_test)
dtr_mse = mean_squared_error(dt_pred,y_test)

from sklearn.ensemble import RandomForestRegressor
rt = RandomForestRegressor()
rt.fit(x_train,y_train)
rt_pred = rt.predict(x_test)
rt_acc = r2_score(rt_pred,y_test)
rt_mse = mean_squared_error(rt_pred,y_test)

from sklearn.model_selection import GridSearchCV,cross_val_score
params = {'fit_intercept': [True, False],'copy_X': [True, False]}
grid = GridSearchCV(param_grid = params,estimator = le,cv=5, scoring='neg_mean_squared_error')
grid.fit(x_train,y_train)
print("Best Parameters:", grid.best_params_)
print("Best Negative MSE Score:", grid.best_score_)

accuracy = cross_val_score(estimator = le,X = x_train,y = y_train,cv = 10,scoring='neg_mean_squared_error')
mse = -accuracy

final_model = LinearRegression(fit_intercept= False,copy_X= True)
final_model.fit(x_train,y_train)

with open('Heightweight.pkl','wb') as file:
    pickle.dump(final_model,file)

import os
os.path.abspath('Heightweight.pkl')
