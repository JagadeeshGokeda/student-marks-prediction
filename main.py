import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
df = pd.read_csv("StudentPerformanceFactors.csv")
df = df[['Hours_Studied', 'Attendance',
       'Previous_Scores','Exam_Score']]
df = df.dropna()
df = df.drop_duplicates()
x = df[['Hours_Studied', 'Attendance',
       'Previous_Scores']]
y = df["Exam_Score"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(x_train)
x_test = poly.transform(x_test)
model = Ridge(alpha=1.0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(f"mean squared error is {mse}")
print(f"r2 is {r2}")
print(f"rmse is {rmse}")
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print(results.head())

