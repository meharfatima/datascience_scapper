# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:47:03 2021

@author: FatimMe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('salary_data_eda.csv')

#select columns
df_mod = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector','Revenue', 'num_comp', 'employer_provided_salary', 'per_hour', 'state', 'same_state', 'age_of_company', 'python', 'spark', 'aws', 'excel', 'sql','job_simpl', 'seniority', 'desc_len']]

#dummy column
df_dum = pd.get_dummies(df_mod)

#test train split
X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values

from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#building model

#stats old model
import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
model_ols = sm.OLS(y_train,X_sm)
results = model_ols.fit()
results.params
results.summary()

#sklearn linear model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#lasso regression
lm_ls = Lasso()
lm_ls.fit(X_train, y_train)
np.mean(cross_val_score(lm_ls, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=i/100)
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha,error)
err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]
lm_ls = Lasso(alpha=0.09)
lm_ls.fit(X_train, y_train)
#random forest regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
np.mean(cross_val_score(rfr, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#grid search CV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 300, 10), 'max_features':["auto", "sqrt", "log2"], 'criterion': ('mse', 'mae')}
gs = GridSearchCV(rfr, parameters, scoring = 'neg_mean_absolute_error', n_jobs = -1)
gs.fit(X_train, y_train)
gs.best_score_
gs.best_estimator_

#predict test
t_pred_lm = lm.predict(x_test)
t_pred_lml = lm_ls.predict(x_test)
t_pred_rfr = gs.best_estimator_.predict(x_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, t_pred_lm)
mean_absolute_error(y_test, t_pred_lml)
mean_absolute_error(y_test, t_pred_rfr)
