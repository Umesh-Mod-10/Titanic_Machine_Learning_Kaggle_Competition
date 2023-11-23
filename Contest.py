# %% Importing the necessary library:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# %% Getting the datasets:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/train_Titanic.csv")
test = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/test_Titanic.csv")

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

# %% Filling the Nan values in Age column:
# region By SimpleImputer:

impute = data.isna().sum()
Impute = SimpleImputer()
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

# %% Changing the Categorical to numbers:

label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])
test['Sex'] = label.fit_transform(test['Sex'])
test['Embarked'] = label.fit_transform(test['Embarked'])

# %% Dropping the Name value:

data.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

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

# Fit train data to GBC
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
gbc.fit(X_train, y_train.ravel())
Y_predict = gbc.predict(X_test)

# Random Forest:
Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Y_prediction = np.array(test['PassengerId'])
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)

# %% Compling the Format:

Final = pd.DataFrame([Y_prediction, Y_predict]).T
Final.columns = ['PassengerId', 'Survived']

# %% Converting our findings into a CSV file:

Final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)

