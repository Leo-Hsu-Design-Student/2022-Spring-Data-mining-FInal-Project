import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from NTU_data import NTU_data

NTU_test_x, NTU_test_y, NTU_test_y_dummy = NTU_data()

def decision_tree(x_train1, x_test1, y_train1, y_test1, f_n, c_n, NTU):
    best_score = 0
    max_d = 0
    for i in range(1, 11):
        dec_tree = DecisionTreeClassifier(max_depth = i, random_state = 1)
        dec_tree.fit(x_train1, y_train1)
        y_prediction = dec_tree.predict(x_test1)
        score = dec_tree.score(x_test1, y_test1)
        if( score > best_score):
            best_score = score
            max_d = i
            dec_tree_best = dec_tree

    print(f"decision tree scores: {dec_tree_best.score(x_test1, y_test1)}", f"depth = {max_d}")

    #store the decision tree figure
    plt.figure()
    plot_tree(dec_tree_best, filled = True, feature_names = f_n, class_names= c_n)
    plt.savefig('decision_tree.png')

    #testing set of NTU
    if NTU:
        NTU_pred_y = dec_tree_best.predict(NTU_test_x)
        score_ = dec_tree_best.score(NTU_test_x, NTU_test_y_dummy)
        print(f"NTU decision tree testing score = {score_}")

    # manipulation of covariance matrix
    cm = confusion_matrix(y_test1, y_prediction)
    print(cm, '\n')



def support_vector_machine(x_train1, x_test1, y_train1, y_test1, NTU):
    svm = SVC()
    svm.fit(x_train1, y_train1)
    y_prediction = svm.predict(x_test1)
    print(f"SVM score: {accuracy_score(y_test1, y_prediction)}")

    #testing set of NTU
    if NTU:
        NTU_pred_y = svm.predict(NTU_test_x)
        score_ = svm.score(NTU_test_x, NTU_test_y_dummy)
        print(f"NTU SVM testing score = {score_}")

    cm = confusion_matrix(y_test1, y_prediction)
    print(cm, '\n')

    

def boosting(x_train1, x_test1, y_train1, y_test1, f_n, c_n, NTU):
        adaB = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.3, random_state = 1)
        adaB.fit(x_train1, y_train1)
        y_prediction = adaB.predict(x_test1)
        score = accuracy_score(y_test1, y_prediction)
        print(f"boosting score: {score}")

        #testing set of NTU
        if NTU:
            NTU_pred_y = adaB.predict(NTU_test_x)
            score_ = adaB.score(NTU_test_x, NTU_test_y_dummy)
            print(f"NTU adaB testing score = {score_}")

        cm = confusion_matrix(y_test1, y_prediction)
        print(cm, '\n')