import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


#Read CSV files
X = pd.read_csv(r"C:\CSV\train.csv")
X_test = pd.read_csv(r"C:\CSV\test.csv")


#Drop rows with missing target
X.dropna(axis=0, subset=["Survived"], inplace=True)
y = X.Survived
X.drop(["Survived"], axis=1, inplace=True)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch",]
X = X[features]
X_test = X_test[features]
X_test_full = X_test[features]
print(X.isnull().sum(axis=0)) #Show which columns have null values

#Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#Get columns with low cardinality
LC_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

#Get numeric columns
N_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# #Impute numeric missing values with mean using
X_train_full["Age"].fillna(X_train_full["Age"].mean(), inplace=True)
X_valid_full["Age"].fillna(X_valid_full["Age"].mean(), inplace=True)
X_test["Age"].fillna(X_test["Age"].mean(), inplace=True)


#Keep selected columns in data set
columns = LC_cols + N_cols
X_train = X_train_full[columns].copy()
X_valid = X_valid_full[columns].copy()
X_test = X_test[columns]
print(X_test)

#One-hot encode data
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
X_train, X_test = X_train.align(X_test, join="left", axis=1)

#Define and fit model
model = XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state=0)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
predictions_test = model.predict(X_test)

#Mean Absolute Error
MAE = mean_absolute_error(y_valid, predictions)
print("Mean Absolute Error:", MAE)

submission = pd.DataFrame({ 'PassengerId': range(892,1310,1),
                            'Survived': predictions_test })
submission.to_csv("submission.csv", index=False)
