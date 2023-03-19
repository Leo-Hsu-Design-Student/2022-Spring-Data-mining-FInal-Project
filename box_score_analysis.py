from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import numpy as np
import statsmodels.api as sm

from linear_model import *
from classifier import *


#read the box score
games = pd.read_csv('box2005to2022')
total_per_game = games.loc[games['Unnamed: 0_level_0'] == 'School Totals'].iloc[:, 1:]

x = total_per_game.iloc[:, :-1]
x.columns = games.iloc[0, 1:-1]
del x['MP']
print(x)
y = total_per_game.iloc[:, -1]
print(x.columns)

#make the binary ouput y
y_dummy = y.to_numpy().astype(int)
y_dummy[y_dummy > 0] = 1
y_dummy[y_dummy < 0] = 0
y_dummy = pd.Series(y_dummy)

#cross validation set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=1)
#binary y
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x, y_dummy, test_size=0.1, random_state=1)

#OLS for all data 
ols = sm.OLS(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
ols_fit = ols.fit()
print(ols_fit.summary())

#compare all data in LinearRegression() with OLS
ols2 = lm.LinearRegression()
ols2_fit = ols2.fit(x, y)

f_n = ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
c_n = ['Lose', 'Win']
NTU = True

least_square_regression(x_train, x_test, y_train, y_test, f_n, NTU)
Lasso_regression(x_train, x_test, y_train, y_test, f_n, NTU)
decision_tree(x_train1, x_test1, y_train1, y_test1, f_n, c_n, NTU)
support_vector_machine(x_train1, x_test1, y_train1, y_test1, NTU)
boosting(x_train1, x_test1, y_train1, y_test1, f_n, c_n, NTU)


