import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

#Import Data
df = pd.read_csv("train.csv")
df = df.replace(np.nan, df['Age'].mean(), regex=True)
#df = df.fillna(value=-1)
#Parsing for Titles

names = df['Name']
titles = []

for i in range(len(names)):
	s = names[i]
	title = re.search(', (.*)\.', s)
	title = title.group(1)
	titles.append(title)

df['Titles'] = titles

#Removing columns
df = df.drop(['PassengerId'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.drop(['Cabin'], axis=1)

#Converting catagorical data to numerical data
lb = preprocessing.LabelEncoder()

df['Sex_C'] = lb.fit_transform(df['Sex'])
df['Ticket_C'] = lb.fit_transform(df['Ticket'])
df['Embarked_C'] = lb.fit_transform(df['Embarked'])
df['Titles_C'] = lb.fit_transform(df['Titles'])

df = df.drop(['Sex'], axis=1)
df = df.drop(['Ticket'], axis=1)
df = df.drop(['Embarked'], axis=1)
df = df.drop(['Titles'], axis=1)

oh = preprocessing.OneHotEncoder(sparse = False)

df_x = df.drop(['Survived'], axis = 1)
df_y = df[['Survived']]

df_x_trans = oh.fit_transform(df_x)

x_train, x_test, y_train, y_test = train_test_split(df_x_trans, df_y, test_size = 0.15, random_state = 10)

clf = linear_model.LogisticRegression()
clf = clf.fit(x_train, y_train)

a = clf.predict(x_test)

print(accuracy_score(y_test, a))
