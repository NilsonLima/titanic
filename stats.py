import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train/train.csv")
print(train["Survived"].value_counts( ))
# print(train["Pclass"].value_counts( ))
# print(train["Sex"].value_counts( ))
# print(train["Age"].value_counts( ))
# print(train["Fare"].value_counts( ))
# print(train["SibSp"].value_counts( ))
# print(train["Parch"].value_counts( ))
# print(train["Ticket"].value_counts( ))

# print(train.loc[pd.isnull(train["Cabin"]) == True, "Survived"].value_counts( ))
# print(pd.isnull(train["Cabin"]).value_counts( ))

# print(train.loc[train["Embarked"] == 'S', "Survived"].value_counts( ))
# print(train.loc[train["Embarked"] == 'C', "Survived"].value_counts( ))
# print(train.loc[train["Embarked"] == 'Q', "Survived"].value_counts( ))
# print(train["Embarked"].value_counts( ))

# print(train.loc[train["Pclass"] == 1, "Fare"].value_counts( ))
# print(train.loc[train["Pclass"] == 2, "Fare"].value_counts( ))
# print(train.loc[train["Pclass"] == 3, "Fare"].value_counts( ))
# print(train["Pclass"].value_counts( ))

# print(train.loc[(pd.isnull(train["Cabin"]) == True) & (train["Survived"] == 0), "Sex"].value_counts( ))
# print(train["Survived"].value_counts())

# print("Southampton port...")
# print(train.loc[train["Embarked"] == 'S', "Ticket"].value_counts( ))

# print("Cherbourg port...")
# print(train.loc[train["Embarked"] == 'C', "Ticket"].value_counts( ))

# print("Queenstown port...")
# print(train.loc[train["Embarked"] == 'Q', "Ticket"].value_counts( ))

# print(train.loc[train.Survived == 1, "Cabin"].value_counts( ))
