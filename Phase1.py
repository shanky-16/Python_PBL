import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
plt.figure()
plt.hist(data["salary"], bins=20)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

#Scatter Plots
plt.figure()
plt.scatter(data["years_experience"], data["salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.figure()
plt.scatter(data["leadership_score"], data["salary"])
plt.title("Leadership vs Salary")
plt.xlabel("Leadership Score")
plt.ylabel("Salary")
plt.show()

#Correlation Matrix
plt.figure(figsize=(10,6))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=False)
plt.title("Correlation Matrix")
plt.show()

#Outlier Detection
plt.figure()
plt.boxplot(data["salary"])
plt.title("Before Outlier Removal")
plt.show()

#Outlier Removal (IQR)
Q1 = data["salary"].quantile(0.25)
Q3 = data["salary"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data = data[(data["salary"] >= lower) & (data["salary"] <= upper)]

#After Outlier Removal
plt.figure()
plt.boxplot(data["salary"])
plt.title("After Outlier Removal")
plt.show()

#Encode categorical data
le = LabelEncoder()

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

#Split data
X = data.drop("salary", axis=1)
y = data["salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)