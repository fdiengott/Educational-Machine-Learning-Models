import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


dataframe = pd.read_csv('data/train.csv') # or read_html, read_json, read_sql, etc.

# a list of the categorical columns that need to be one-hot encoded*
categorical_columns = [...]

dataframe - pd.get_dummies(data=dataframe, prefix=categorical_columns, columns=categorical_columns)

features = dataframe.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(dataframe[features], dataframe['target'], train_size=0.8)

# testing the best minimum samples to list, max depth, and n estimators (forest size)
min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10, 50, 100, 200]

accuracies_train = []
accuracies_test = []

kwarg_list = ['min_samples_split', 'max_depth', 'n_estimators']
list_list = [min_samples_split_list, max_depth_list, n_estimators_list]

# : this tests each parameter above independent of the others.
# Ideally all combinations would be tested, but I'm keeping it simple for now.
for kwarg, list in zip(kwarg_list, list_list):
    accuracies_train_local = []
    accuracies_test_local = []
    kwargs = {}
    for item in list:
        kwargs[kwarg] = item
        model = RandomForestClassifier(**kwargs).fit(X_train, y_train)

        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)
        accuracies_train_local.append(accuracy_score(predictions_train, y_train))
        accuracies_test_local = .append(accuracy_score(predictions_test, y_test))

    accuracies_train.append(accuracies_train_local)
    accuracies_test.append(accuracies_test_local)

# going through the above accuracies would find the best combination of parameters.


# ******************************************************************
# ******************************************************************


# XGBoost

# I'm gonna split the training data into 80% training and 20% cross-validation
n = int(len(X_train) * 0.8)
X_train_fit, X_train_eval = X_train[:n], X_train[n:]
y_train_fit, y_train_eval = y_train[:n], y_train[n:]

xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1, verbosity=1)
xgb.fit(X_train_fit, X_train_eval, eval_set=[(X_test_eval, y_train_eval)], early_stopping_rounds=20)

print(xgb_model.best_iteration)

predictions_train = xgb_model.predict(X_train)
predictions_test = xgb_model.predict(X_test)

accuracy_train = accuracy_score(predictions_train, y_train)
accuracy_test = accuracy_score(predictions_test, y_test)






# * one-hot-encoding: instead of a column having more than 2 values,
# split into several feature columns, which each accept a 1 or 0
