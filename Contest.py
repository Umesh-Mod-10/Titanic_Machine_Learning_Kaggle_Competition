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
from sklearn.metrics import accuracy_score

# %% Getting the datasets:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/train_Titanic.csv")
test = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/test_Titanic.csv")
sol = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/titanic_sol.csv")

# %% Getting the basic details of train:

stats_data = data.describe()
test_data = test.describe()
print(data.info())
print(test.info())
print(data.isna().sum())
print()
print(test.isna().sum())

# %% Remove the Cabin column as it has a lot of Nan values:

data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

# %% Remove the Nan from Embarked:

#data.dropna(subset=['Embarked'], inplace=True)


# %% Check for the duplicate values:

print(data.duplicated().sum())


# %%

impute = test.isna().sum()
print(impute[impute != 0])

# %% Filling the Nan values in Age column:

# region By SimpleImputer:

impute = data.isna().sum()
Impute = SimpleImputer(strategy="median")
test['Fare'] = Impute.fit_transform(test['Fare'].to_numpy().reshape((-1, 1)))
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# endregion

# region By Feature Engineering:

data['Title'] = data['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
print(data['Title'].value_counts())
sn.histplot(data=data, kde=True, x='Age')
plt.show()
data['Age'].fillna(data.groupby('Title')['Age'].transform("mean"), inplace=True)
test['Title'] = test['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
sn.histplot(data=data, kde=True, x='Age')
plt.show()
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

data = data.astype("float64")
test = test.astype("float64")

# %% Splitting the Dependent and Independent:

X_train = data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = data.loc[:, 'Survived'].to_numpy().reshape((-1, 1))
X_test = test.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_test = sol.loc[:, 'Survived'].to_numpy().reshape((-1, 1))

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
print(accuracy_score(y_test,Y_predict))

# %%
# Fit train data to GBC
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
gbc.fit(X_train, y_train.ravel())
Y_predict = gbc.predict(X_test)
gbc1 = gbc.score(X_train, y_train)
print(accuracy_score(y_test,Y_predict))

# %%
# Random Forest:

Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
rf1 = Rf.score(X_train, y_train)
print(accuracy_score(y_test, Y_predict))

# %%
xgb_r = xg.XGBClassifier(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)
xbr1 = xgb_r.score(X_train, y_train)
print(accuracy_score(y_test, Y_predict))

# %%

models = pd.DataFrame({
    'Model': ['Logistic Regression',
              'Random Forest', 'XGB', 'Gradient Booster'],
    'Score': [lr1, rf1, xbr1, gbc1]})
sorted_model = models.sort_values(by='Score', ascending=False)

fig = plt.figure(figsize=(13, 10))
sn.barplot((sorted_model['Score']), color='lightgreen', hatch='/', edgecolor='black', alpha=0.6)
plt.xticks(ticks=range(len(sorted_model['Score'])), labels=sorted_model['Model'])
plt.grid()
plt.show()

# %% Compling the Format:

Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
print(accuracy_score(y_test,Y_predict))
Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']
Final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)
Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']

# %%

Final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)

# %%
