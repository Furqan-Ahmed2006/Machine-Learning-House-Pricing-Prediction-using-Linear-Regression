import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
df=pd.read_excel("House-Data.xlsx")
x=df.drop("SalePrice",axis=1)
y=df["SalePrice"]

#----------------Training and Testing Phase-----------------------------

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("-----------------Testing of X-Label----------------------")
print(x_test)
print("----------------------Training of X-Label---------------")
print(x_train)
print("-----------------Testing of Y-Label----------------------")
print(y_test)
print("----------------------Training of Y-Label---------------")
print(y_train)

# ------------------------Scalling Phase-------------------------------


min_max=MinMaxScaler()
x_train=min_max.fit_transform(x_train)
x_test=min_max.transform(x_test)

scaled_train_df = pd.DataFrame(x_train, columns=x.columns)
scaled_test_df = pd.DataFrame(x_test, columns=x.columns)

print("------------------Scalling of Data------------------------")
print("Scaled Training Data (first 2 columns):")
print(scaled_train_df[["LotArea", "GarageCars"]].head())
print("\nScaled Test Data (first 2 columns):")
print(scaled_test_df[["LotArea", "GarageCars"]].head())

#------------------------Model Prediction----------------------------

model=LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("-------------------------Prediction---------------------------------")
print(y_pred)

#----------------------Evaluation-------------------------------

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("----------------Model Evaluation-----------------------")
print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"RMSE:{rmse:.2f}")
print(f"R2:{r2:.2f}")

# ----------------------Predictions----------------------------------------
print("--------------------------House Price  Prediction----------------------")
a=int(input("Enter the plot Area:"))
b=int(input("Enter the quality rate out of 10:"))
c=int(input("Enter the Year in which it builts: "))
d=int(input("Enter Total Basement Area:"))
e=int(input("Enter Ground Living Area: "))
f=int(input("Enter the Garrage Cars:"))
new_house = pd.DataFrame([[a,b,c,d,e,f]],columns=x.columns)  # example input
new_house_scaled = min_max.transform(new_house)
price = model.predict(new_house_scaled)[0]
print(f"Predicted Price: pkr{price:.2f}")

# Graphical View with Prediction Line
plt.figure(figsize=(10,6))
plt.plot([y_test.min(), y_test.max()],[y_test.min(),y_test.max()], 'red')
plt.scatter(y_test,y_pred,alpha=0.7,color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Price")
plt.title("Actual Vs Predicted Price")
plt.grid(True)
plt.show()

errors=y_test-y_pred
plt.hist(errors,bins=20,color="purple",edgecolor="black",alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Distribution of Predicted Errors")
plt.show()