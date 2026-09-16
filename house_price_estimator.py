import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Dataset [Square Feet, Bedrooms] -> Price (in $1000s)
X = np.array([
    [500, 1],
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 5]
])
y = np.array([50, 100, 150, 200, 250])

# 2. Train Model
model = LinearRegression()
model.fit(X, y)

# 3. Estimate Price for 1200 SqFt & 2 Bedrooms
new_house = [[1200, 2]]
price = model.predict(new_house)

print(f"Estimated Price: ${price[0]:.2f}k")
