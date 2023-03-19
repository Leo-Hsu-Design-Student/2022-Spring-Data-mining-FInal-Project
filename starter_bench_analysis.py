from ast import Not
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.impute import SimpleImputer

from linear_model import *
from classifier import *

f_n = ['MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
f_n1 = []
for stri in f_n:
    element = stri + '_b'
    f_n1.append(element)
print(f_n1)
c_n = ['Lose', 'Win']

#read the box score
games = pd.read_csv('box2005to2022')

# print(games)
total_per_game = games.loc[games['Unnamed: 0_level_0'] == 'School Totals'].iloc[:, 1:]


#store all the "total" row's index
total_row_index = total_per_game.index.tolist()
total_row_index = [0] + total_row_index
length = len(total_row_index)

#Preprocessing starters and bench
starter = pd.DataFrame()
bench = pd.DataFrame()

for i in range(length-1):
    starter_start = total_row_index[i] + 1
    starter_end = starter_start + 5
    starter_a_game = games.iloc[starter_start:starter_end, 1:].astype(float).sum().to_numpy().reshape(1, 22)
    starter_a_game = pd.DataFrame(starter_a_game)
    bench_a_game = games.iloc[starter_end: total_row_index[i+1], 1:].astype(float).sum().to_numpy().reshape(1, 22)
    bench_a_game = pd.DataFrame(bench_a_game)
    starter = pd.concat([starter, starter_a_game], axis = 0)
    bench = pd.concat([bench, bench_a_game], axis = 0)
#make the starter and bench columns
starter.columns = f_n
bench.columns = f_n1
f_n_all = f_n + f_n1

starter.iloc[:, 3] = starter.iloc[:, 1] / starter.iloc[:, 2]
starter.iloc[:, 6] = starter.iloc[:, 4] / starter.iloc[:, 5]
starter.iloc[:, 9] = starter.iloc[:, 7] / starter.iloc[:, 8]
starter.iloc[:, 12] = starter.iloc[:, 10] / starter.iloc[:, 11]
bench.iloc[:, 3] = bench.iloc[:, 1] / bench.iloc[:, 2]
bench.iloc[:, 6] = bench.iloc[:, 4] / bench.iloc[:, 5]
bench.iloc[:, 9] = bench.iloc[:, 7] / bench.iloc[:, 8]
bench.iloc[:, 12] = bench.iloc[:, 10] / bench.iloc[:, 11]

print(starter)
print(bench)

y = total_per_game.iloc[:, -1]
x = pd.concat([starter, bench], axis = 1).round(3)
imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imp.fit(x)
x = pd.DataFrame(imp.transform(x))
x.columns = f_n_all

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

#compare all data in LinearRegression() with OLS
ols2 = lm.LinearRegression()
ols2_fit = ols2.fit(x, y)
NTU = False

least_square_regression(x_train, x_test, y_train, y_test, f_n_all, NTU)
Lasso_regression(x_train, x_test, y_train, y_test, f_n_all, NTU)
decision_tree(x_train1, x_test1, y_train1, y_test1, f_n_all, c_n, NTU)
support_vector_machine(x_train1, x_test1, y_train1, y_test1, NTU)
boosting(x_train1, x_test1, y_train1, y_test1, f_n_all, c_n, NTU)


    




