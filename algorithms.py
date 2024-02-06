import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("talib_indicators.csv", parse_dates=["datetime"], index_col=["datetime"])

indicators = list(df.columns)
indicators.remove("close")

horizon = 96
df["ret"] = np.log(df["close"]).shift(-horizon) - np.log(df["close"])

condlist = [df["ret"] > 0]
choicelist = [1]
df["y"] = np.select(condlist, choicelist, 0)

df_train = df.loc[(df.index >= "2021-01-01 00:00:00") & (df.index < "2023-01-01 00:00:00")]
df_valid = df.loc[(df.index >= "2023-01-01 00:00:00")]

n_train = len(df_train)
df_pos = df_train.loc[df_train["y"] == 1]
df_neg = df_train.loc[df_train["y"] == 0]

df_train = pd.concat([
    df_pos.sample(n_train // 4, random_state=42),
    df_neg.sample(n_train // 4, random_state=42),
])



rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=3)
rf.fit(df_train[indicators], df_train["y"])

y_train_pred = rf.predict(df_train[indicators])
train_accuracy = accuracy_score(df_train["y"], y_train_pred)
train_macro_precision = precision_score(df_train["y"], y_train_pred, average="macro")
train_micro_precision = precision_score(df_train["y"], y_train_pred, average="micro")
train_macro_recall = recall_score(df_train["y"], y_train_pred, average="macro")
train_micro_recall = recall_score(df_train["y"], y_train_pred, average="micro")
print("Train Accuracy:", train_accuracy)
print("Train confusion matrix") 
print(confusion_matrix(df_train["y"], y_train_pred))
print("Train Macro Precision:", train_macro_precision)
print("Train Micro Precision:", train_micro_precision)
print("Train Macro Recall:", train_macro_recall)
print("Train Micro Recall:", train_micro_recall)


y_valid_pred = rf.predict(df_valid[indicators])
valid_accuracy = accuracy_score(df_valid["y"], y_valid_pred)
valid_macro_precision = precision_score(df_valid["y"], y_valid_pred, average="macro")
valid_micro_precision = precision_score(df_valid["y"], y_valid_pred, average="micro")
valid_macro_recall = recall_score(df_valid["y"], y_valid_pred, average="macro")
valid_micro_recall = recall_score(df_valid["y"], y_valid_pred, average="micro")
print("\nValid Accuracy:", valid_accuracy)
print("Valid confusion matrix")
print(confusion_matrix(df_valid["y"], y_valid_pred))
print("Valid Macro Precision:", valid_macro_precision)
print("Valid Micro Precision:", valid_micro_precision)
print("Valid Macro Recall:", valid_macro_recall)
print("Valid Micro Recall:", valid_micro_recall)

for imp, indi in sorted(zip(rf.feature_importances_, indicators), reverse=True):
    print(f"{indi}:      {imp:.4f}")