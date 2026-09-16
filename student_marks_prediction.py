import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Dataset (Study Hours vs Exam Marks)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 45, 50, 60, 68, 75, 82, 90])

# 2. Initialize and Train Model
model = LinearRegression()
model.fit(X, y)

# 3. Predict Marks for 6.5 Hours of Study
test_hours = [[6.5]]
predicted_marks = model.predict(test_hours)

print(f"Predicted Score for {test_hours[0][0]} study hours: {predicted_marks[0]:.2f}")
