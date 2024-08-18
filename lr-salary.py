import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Package Explanations:
# 1. `numpy` (np): A powerful library for numerical computing in Python. 
#    It's often used for working with arrays and performing mathematical operations on them.
# 2. `pandas` (pd): A data manipulation and analysis library. It provides data structures like 
#    DataFrame and Series, which are perfect for handling tabular data.
# 3. `scikit-learn` (`sklearn`): A library for machine learning in Python. It offers simple and efficient tools 
#    for data mining and data analysis, including various machine learning algorithms like linear regression.
# 4. `matplotlib.pyplot` (plt): A plotting library used for creating static, animated, and interactive visualizations 
#    in Python. It's very handy for creating charts and plots.

# Step 3: Prepare the Data
# Create a DataFrame with the data
# `pandas.DataFrame` is used to create a tabular structure to hold the data.
data = {
    'Years_of_Experience': [1, 2, 3, 4, 5],
    'Salary': [50, 55, 60, 65, 70]
}
df = pd.DataFrame(data)

# Step 4: Define the Independent and Dependent Variables
# `X` is the independent variable (Years of Experience) and `y` is the dependent variable (Salary).
# `df[['Years_of_Experience']]` ensures `X` is a 2D array (a DataFrame with one column).
X = df[['Years_of_Experience']]  # Independent variable (must be a 2D array)
y = df['Salary']                # Dependent variable

# Step 5: Create the Linear Regression Model
# `LinearRegression()` is a class from `scikit-learn` that provides the linear regression algorithm.
model = LinearRegression()

# Step 6: Fit the Model
# `fit(X, y)` is a method that trains the linear regression model using the provided data.
# It computes the best-fit line for the given data.
model.fit(X, y)

# Step 7: Get the Model Parameters
# The intercept and slope of the line are stored in `model.intercept_` and `model.coef_` respectively.
# `intercept_` gives the Y-intercept (where the line crosses the Y-axis when X=0).
# `coef_` gives the slope (how much Y changes for a one-unit change in X).
intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept}")
print(f"Slope: {slope}")

# Step 8: Make Predictions
# `predict()` is used to predict new values based on the trained model.
# Here, we're predicting the salary for 6 years of experience.
years_of_experience = 6
predicted_salary = model.predict([[years_of_experience]])
print(f"Predicted salary for {years_of_experience} years of experience: ${predicted_salary[0] * 1000:.2f}")

# Optional: Plot the Results
# The scatter plot shows the actual data points (blue dots).
# The line plot shows the best-fit line predicted by the model (red line).
# This helps visualize the relationship between years of experience and salary.
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (in $1000s)')
plt.title('Linear Regression: Salary vs Years of Experience')
plt.show()