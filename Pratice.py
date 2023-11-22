# %% Importing the necessary library:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xg

# %% Getting the datasets:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/train_Titanic.csv")
test = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/test_Titanic.csv")
sol = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/titanic_sol.csv")

# %% Getting the basic details of train:

stats = data.describe()
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

# %% Getting the data of Name much detailed:

data['title'] = data['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
print(data['title'].value_counts())
print(pd.crosstab(data['title'], data['Sex']))

# %%

data['title'].replace('Ms', 'Miss', inplace=True)
data['title'].replace('Mlle', 'Miss', inplace=True)
data['title'].replace('Mme', 'Mrs', inplace=True)
data['title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Countess', 'Capt', 'Don', 'Jonkheer'], 'Random', inplace=True)
print(data['title'].value_counts())

test['title'] = test['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
print(test['title'].value_counts())
test['title'].replace('Ms', 'Miss', inplace=True)
test['title'].replace('Mlle', 'Miss', inplace=True)
test['title'].replace('Mme', 'Mrs', inplace=True)
test['title'].replace('Sir', 'Mr', inplace=True)
test['title'].replace('Lady', 'Mrs', inplace=True)
test['title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Countess', 'Capt', 'Don', 'Jonkheer'], 'Random', inplace=True)

# %% Filling the Nan values in Age column:

# region By SimpleImputer:

impute = data.isna().sum()
Impute = SimpleImputer(strategy="median")
data['Fare'] = Impute.fit_transform(data['Fare'].to_numpy().reshape((-1, 1)))
test['Fare'] = Impute.fit_transform(test['Fare'].to_numpy().reshape((-1, 1)))
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# endregion

# region By Feature Engineering:

data['Name'] = data['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
sn.histplot(data=data, kde=True, x='Age')
plt.show()
data['Age'].fillna(data.groupby('Name')['Age'].transform("median"), inplace=True)

test['Name'] = test['Name'].str.extract(' ([A-Z,a-z]+)\. ', expand=False)
sn.histplot(data=data, kde=True, x='Age')
plt.show()
test['Age'].fillna(data.groupby('Name')['Age'].transform("median"), inplace=True)
print(data.isna().sum())
print(test.isna().sum())

# endregion
# %% Changing the Age column to int:

data['Age'] = data['Age'].round()
print(pd.cut(data['Age'], bins=4))
test['Age'] = test['Age'].round()
print(pd.cut(test['Age'], bins=4))

# %% Changing the Continuous data into Categorical Data:

bins1 = [0, 20.0, 32.0, 48.0, 64.0, 80.0, float('inf')]
labels1 = [0, 1, 2, 3, 4, 5]
data['Age'] = pd.cut(data['Age'], bins=bins1, labels=labels1, right=False)

bins2 = [0, 15.586, 30.752, 45.918, 61.084, 76.25, float('inf')]
labels2 = [0, 1, 2, 3, 4, 5]
test['Age'] = pd.cut(test['Age'], bins=bins2, labels=labels2, right=False)

# %% Fare Round off Limits:

print(pd.cut(data['Fare'], bins=4))
print(pd.cut(test['Fare'], bins=4))

# %% Converting Fare value:

bins = [-0.512, 128.082, 256.165, 384.247, 512.33]
labels = [0, 1, 2, 3]
data['Fare'] = pd.cut(data['Fare'], bins=bins, labels=labels)

bins = [-0.512, 128.082, 256.165, 384.247, 512.33]
labels = [0, 1, 2, 3]
test['Fare'] = pd.cut(test['Fare'], bins=bins, labels=labels)

# %% Changing the Categorical to numbers:

label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])
test['Sex'] = label.fit_transform(test['Sex'])
test['Embarked'] = label.fit_transform(test['Embarked'])
data['title'] = label.fit_transform(data['title'])
test['title'] = label.fit_transform(test['title'])

# %% Getting unique value of alone:

data['IsAlone'] = data['Parch'] + data['SibSp']
data['IsAlone'] = np.where(data['IsAlone'] > 0, 1, data['IsAlone'])

test['IsAlone'] = test['Parch'] + test['SibSp']
test['IsAlone'] = np.where(test['IsAlone'] > 0, 1, test['IsAlone'])

# %% Dropping the Name value:

data.drop(['Name'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)

# %% Converting all values to float:

data = data.astype(np.float64)
test = test.astype(np.float64)

# %% Splitting the Dependent and Independent:

X_train = data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'IsAlone', 'Fare', 'Embarked', 'title']]
y_train = data.loc[:, 'Survived'].to_numpy().reshape((-1, 1))
X_test = test.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'IsAlone', 'Fare', 'Embarked', 'title']]

print(X_train.info())
print(X_train.info())

# %% Model for ML regression:
# Logistic Regression

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train.ravel())
Y_predict = lr.predict(X_test)
Y_predict = np.array(Y_predict)
Y_prediction = np.array(test['PassengerId'])
Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']
Y_predict = pd.DataFrame(Y_predict)
print(lr.score(X_train, y_train))

# %%
# Fit train data to GBC

gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
gbc.fit(X_train, y_train.ravel())
Y_predict = gbc.predict(X_test)
Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']
print(gbc.score(X_train, y_train))

# %%
# Random Forest:

Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']
print(Rf.score(X_train, y_train))

# %%
xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)
print(xgb_r.score(X_train, y_train))

# %% Compling the Format:

Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']

# %%

Final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)

# %%
