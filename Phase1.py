import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("employee_salary_dataset.csv")
print(data.head())
print(data.info())
print(data.describe())

#Check Missing Values
print(data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)

#Feature Distributions


data.hist(figsize=(15,10), bins=15)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

#Salary Distribution
