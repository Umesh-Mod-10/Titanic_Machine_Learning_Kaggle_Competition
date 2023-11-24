# %% Importing the necessary library:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost as xg
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# %% Getting the datasets:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/train_Titanic.csv")
test = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/test_Titanic.csv")

# %% Getting the basic details of train:

stats_data = data.describe()
stats_test = test.describe()
print(data.info())
print(data.isna().sum())
print(test.isna().sum())

# %% Remove the Cabin column as it has a lot of Nan values:

data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

# %% Check for the duplicate values:

print(data.duplicated().sum())

# %% Filling the Nan values in Age column:
# By SimpleImputer:

impute = data.isna().sum()
Impute = SimpleImputer(strategy="median")
test['Fare'] = Impute.fit_transform(test['Fare'].to_numpy().reshape((-1, 1)))
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# By Feature Engineering:

data['Title'] = data['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
data['Age'].fillna(data.groupby('Title')['Age'].transform("median"), inplace=True)

test['Title'] = test['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
test['Age'].fillna(data.groupby('Title')['Age'].transform("median"), inplace=True)

print(data.isna().sum())
print(test.isna().sum())

# %% Changing the Categorical to numbers:

label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])
test['Sex'] = label.fit_transform(test['Sex'])
test['Embarked'] = label.fit_transform(test['Embarked'])

# %% Dropping the Name value:

data.drop(['Name', 'Title'], axis=1, inplace=True)
test.drop(['Name', 'Title'], axis=1, inplace=True)

# %% Converting all values to float:

data.astype("float64")
test.astype("float64")

# %% Splitting the Dependent and Independent:

X_train = data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = data.loc[:, 'Survived'].to_numpy().reshape((-1, 1))
X_test = test.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# %% Scaling the values:

minmax = MinMaxScaler()
X_train.loc[:, ['Age', 'Fare']] = minmax.fit_transform(X_train.loc[:, ['Age', 'Fare']])
X_test.loc[:, ['Age', 'Fare']] = minmax.fit_transform(X_test.loc[:, ['Age', 'Fare']])

# %% Model for ML regression:
# Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train.ravel())
Y_predict = lr.predict(X_test)
Y_predict = np.array(Y_predict)
Y_prediction = np.array(test['PassengerId'])
lr1 = lr.score(X_train, y_train)

# Fit train data to GBC
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
gbc.fit(X_train, y_train.ravel())
Y_predict = gbc.predict(X_test)
gbc1 = gbc.score(X_train, y_train)

# Random Forest:
Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
rf1 = Rf.score(X_train, y_train)

# %% XGB Regressor:

xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)
xbr1 = xgb_r.score(X_train, y_train)

# %% Tabulation of the results:

models = pd.DataFrame({'Model': ['Logistic Regression','Random Forest', 'XGB', 'Gradient Booster'], 'Score': [lr1, rf1, xbr1, gbc1]})
sorted_model = models.sort_values(by='Score', ascending=False)

# %% The graphical representation of the accuracy score:

fig = plt.figure(figsize=(13, 10))
sn.barplot((sorted_model['Score']), color='lightgreen', hatch='/', edgecolor='black', alpha=0.6)
plt.xticks(ticks=range(len(sorted_model['Score'])), labels=sorted_model['Model'])
plt.grid()
plt.show()

# %% Compling the Format:

Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']

# %% Converting our findings into a CSV file:

Final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)

# %%
