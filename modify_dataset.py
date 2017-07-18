import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor

def fillAgeGaps( ):
	global df

	features = list(filter(lambda x: x not in ["Survived", "Age", "PassengerId"], list(df)))
	print(features)

	X = df[pd.isnull(df.Age) == False][features]
	y = df[pd.isnull(df.Age) == False].Age
	pred_X = df[pd.isnull(df.Age) == True][features]

	clf = RandomForestRegressor(n_estimators=2000, max_features='sqrt', n_jobs=-1)
	clf.fit(X, y)

	df.loc[pd.isnull(df.Age) == True, "Age"] = clf.predict(pred_X)

	return

def dummyFeature(feature):
	global df

	df_dummies = pd.get_dummies(df[feature])
	df_dummies = df_dummies.rename(columns=lambda x: feature + "_" + str(x))
	df = pd.concat([df, df_dummies], axis=1).drop(feature, axis=1)

	return

def mapSex( ):
	global df

	df.Sex = df.Sex.map({'male': 0, 'female': 1})

	return

def extractTicketInfo( ):
	global df

	df.Ticket = df.Ticket.map(lambda x: re.sub('[./]', '', x.upper( )).split( )[0])
	df.ix[df.Ticket.isin(["SOTONOQ", "SOTONO2", "STONO", "STONOQ", "STONO2", "CASOTON"]), "Ticket"] = "SOTON"
	df.ix[df.Ticket.str.isnumeric( ) == True, "Ticket"] = "N"

	dummyFeature("Ticket")

	return

def extractCabinInfo( ):
	global df

	df.ix[pd.isnull(df.Cabin) == True, "Cabin"] = "M"
	df.Cabin = df.Cabin.map(lambda x: str(x)[0])

	dummyFeature("Cabin")

	return

def extractTitleInfo( ):
	global df

	df["Title"] = df.Name.map(lambda x: str(x).split(",")[1].split(".")[0].strip( ))

	Title_Dictionary = {
                        "Capt":       "Rare",
                        "Col":        "Rare",
                        "Major":      "Rare",
                        "Jonkheer":   "Rare",
                        "Don":        "Rare",
                        "Sir" :       "Rare",
                        "Dr":         "Rare",
                        "Rev":        "Rare",
                        "the Countess":"Rare",
                        "Dona":       "Rare",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Rare",
                        "Lady" :      "Rare"
                        }

	df.Title = df.Title.map(Title_Dictionary)
	dummyFeature("Title")

	df = df.drop("Name", axis=1)

	return

def discreteEmbarked( ):
	global df

	df.ix[pd.isnull(df.Embarked) == True, "Embarked"] = "S"
	dummyFeature("Embarked")

	return

def fillFareGaps( ):
	global df

	df.loc[pd.isnull(df.Fare) == True, "Fare"] = df.Fare.mean( )

	return

def familyMembers( ):
	global df

	members = df.SibSp + df.Parch
	df["Family"] = members.map(lambda x: 'S' if x < 2 else 'M' if x <= 4 else 'L')
	dummyFeature("Family")

	return

def familySize( ):
	global df

	members = df.SibSp + df.Parch
	df["FamilySize"] = members.map(lambda x: x + 1)

	return

def printFiles( ):
	global df

	train_frame = df.loc[pd.isnull(df.Survived) == False]
	test_frame = df.loc[pd.isnull(df.Survived) == True]

	test_frame = test_frame.drop("Survived", axis=1)

	train_frame.to_csv("train/train_modified.csv", index=False)
	test_frame.to_csv("test/test_modified.csv", index=False)

	return

def main( ):
	global df

	discreteEmbarked( )
	mapSex( )
	df = df.drop("Ticket", axis=1) #extractTicketInfo( )
	extractCabinInfo( )
	extractTitleInfo( )
	dummyFeature("Pclass")
	fillFareGaps( )
	familyMembers( )
	#familySize( )
	fillAgeGaps( )
	printFiles( )

	return

train = pd.read_csv("train/train.csv")
test = pd.read_csv("test/test.csv")
df = pd.concat([train, test])

main( )
