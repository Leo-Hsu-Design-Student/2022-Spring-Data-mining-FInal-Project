import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import pickle

from NTU_data import NTU_data
sns.set(font_scale = 0.6)

NTU_test_x, NTU_test_y, NTU_test_y_dummy = NTU_data()

def least_square_regression(x_train1, x_test1, y_train1, y_test1, f_n, NTU):
    lr = lm.LinearRegression(normalize=True)
    linear_m = lr.fit(x_train1, y_train1)
    y_prediction = linear_m.predict(x_test1)

    mse = mean_squared_error(y_test1, y_prediction)
    r_square = r2_score(y_test1, y_prediction)
    print(f"lm's best mse: {mse}")
    print(f"lm's best r2: {r_square}")

    #store the coefficients
    coef = linear_m.coef_
    length = len(x_train1.columns)
    for i in range(length):
        print(f"{f_n[i]}: {coef[i]}")

    print()
    sns.barplot(x = coef, y = f_n, orient = 'h').set(title = 'OLS')
    plt.savefig('linear_coef_nor')
    plt.close()

    # #store the best model
    # with open('multi_lm_pkl', 'wb') as files:
    #     pickle.dump(linear_m, files)

    # with open('multi_lm_pkl', 'rb') as f:
    #     best_lm = pickle.load(f)

    #testing set of NTU
    if NTU:
        NTU_pred_y = linear_m.predict(NTU_test_x)
        mse_ = mean_squared_error(NTU_test_y, NTU_pred_y)
        r_square_ = r2_score(NTU_test_y, NTU_pred_y)
        print(f"NTU lm's test mse: {mse_}")
        print(f"NTU lm's test r2: {r_square_}")

    



def Lasso_regression(x_train1, x_test1, y_train1, y_test1, f_n, NTU):
    lasso = lm.Lasso(alpha = 0.02, normalize=True)
    lasso_m = lasso.fit(x_train1, y_train1)
    y_prediction = lasso_m.predict(x_test1)

    mse = mean_squared_error(y_test1, y_prediction)
    r_square = r2_score(y_test1, y_prediction)
    print(f"Lasso's best mse: {mse}")
    print(f"Lasso's best r2: {r_square}")

    #store the coefficients
    coef = lasso_m.coef_
    length = len(x_train1.columns)
    for i in range(length):
        print(f"{f_n[i]}: {coef[i]}")

    print()
    sns.barplot(x = coef, y = f_n, orient = 'h').set(title = 'Lasso')
    plt.savefig('lasso_coef_nor')
    plt.close()

    # #store the best model
    # with open('lasso_pkl', 'wb') as files:
    #     pickle.dump(lasso_m, files)

    # with open('lasso_pkl', 'rb') as f:
    #     best_lasso = pickle.load(f)

    #testing set of NTU
    if NTU:
        NTU_pred_y = lasso_m.predict(NTU_test_x)
        mse_ = mean_squared_error(NTU_test_y, NTU_pred_y)
        r_square_ = r2_score(NTU_test_y, NTU_pred_y)
        print(f"NTU lasso's test mse: {mse_}")
        print(f"NTU lasso's test r2: {r_square_}")
