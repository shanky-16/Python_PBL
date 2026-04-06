import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nChecking missing values:")
print(df.isnull().sum())

print("\nChecking duplicates:")
print(df.duplicated().sum())

plt.figure()
sns.histplot(df['salary'])
plt.title("Salary Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

df = pd.get_dummies(df, drop_first=True)

X = df.drop("salary", axis=1)
y = df["salary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LinearRegression()

sfs = SequentialFeatureSelector(
    model,
    n_features_to_select=10,
    direction='forward'
)

sfs.fit(X_train, y_train)

X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_sfs, y_train)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sfs, y_train)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_sfs, y_train)

def evaluate(model, name):
    pred = model.predict(X_test_sfs)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    print(f"\n{name} Results:")
    print("R2 Score:", r2)
    print("RMSE:", rmse)

    
    plt.figure()
    plt.scatter(y_test, pred)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"{name} - Actual vs Predicted")
    plt.show()

evaluate(lr, "Linear Regression")
evaluate(ridge, "Ridge Regression")
evaluate(lasso, "Lasso Regression")