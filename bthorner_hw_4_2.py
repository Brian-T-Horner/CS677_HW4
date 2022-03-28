"""
Brian Horner
CS 677 - Summer 2
Date: 8/3/2021
Week 4 Homework Question 2
This program attempts to use a linear, quadratic, cubic and two GLM models
to fit the provided data. It prints graphs of the actual values and
predicted line as well as prints the loss function in a table.

I honestly have no idea why my loss functions are so large. I thought it
would be because of scaling but that does not seem to be the case.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def sse(actual, predict):
    total_value = 0
    for index, value in enumerate(predict):
        current_value = (actual[index] - value)**2
        total_value += current_value

    return round(total_value, 2)

def table_printer(stats_list):
    """Formats the computations for table printing."""
    header_list = ['-- Model --', 'SSE (death_event=0)', 'SSE ('
                                                           'death_event=1)']
    print_list = []
    print("--- Summary of Model Loss Functions for Survived and Passed "
          "DataSets. ---\n")
    for list in stats_list:
        print_list.append(list)
    print_list.insert(0, header_list)
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(20) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))

stats_list = []

# Loading csv into dataframe
heart_df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Splitting dataframe based off death event boolean
df_0 = heart_df.loc[heart_df['DEATH_EVENT'] == 0]
df_1 = heart_df.loc[heart_df['DEATH_EVENT'] == 1]

# Grabbing specific columns of data
df_1 = df_1.loc[:, ['creatinine_phosphokinase', 'platelets']]
df_0 = df_0.loc[:, ['creatinine_phosphokinase', 'platelets']]

df_0_x = df_0['creatinine_phosphokinase'].values
df_0_y = df_0['platelets'].values

df_1_x = df_1['creatinine_phosphokinase'].values
df_1_y = df_1['platelets'].values




# ---------------------------------------------------------------------

""" Linear Model"""
print("--- Linear Model ---")
X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(df_0_x, df_0_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(df_1_x, df_1_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
# Getting linear model weights
linear_weights_0 = np.polyfit(X_train_0, Y_train_0, 1)
linear_weights_1 = np.polyfit(X_train_1, Y_train_1, 1)

print(f"Weights for linear regression model for survived dataset are "
      f"{linear_weights_0}.")
print(f"Weights for linear regression model for passed dataset are "
      f" {linear_weights_1}.")

# Converting weights to model
linear_model_0 = np.poly1d(linear_weights_0)
linear_model_1 = np.poly1d(linear_weights_1)

# Predicting using linear model
linear_predicted_0 = linear_model_0(X_test_0)
linear_predicted_1 = linear_model_1(X_test_1)

# Calculating SSE for Y_test and predicted values of Y
linear_sse_0 = sse(Y_test_0, linear_predicted_0)
linear_sse_1 = sse(Y_test_1, linear_predicted_1)


print(f"Sum of Squared Residuals for survived dataset and a linear "
      f"regression model is {linear_sse_0:,.2f}.")

print(f"Sum of Squared Residuals for passed dataset and a linear "
      f"regression model is {linear_sse_1:,.2f}")

# Adding values into stats_list for printing
temp_list = ['y =ax+b', linear_sse_0, linear_sse_1]
stats_list.append(temp_list)
# -----------------------------------------------------------------------
"""Quadratic"""
print("\n --- Quadratic Model ---")
X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(df_0_x, df_0_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(df_1_x, df_1_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )

# Getting quadratic model weights
quadratic_weights_0 = np.polyfit(X_train_0, Y_train_0, 2)
quadratic_weights_1 = np.polyfit(X_train_1, Y_train_1, 2)

print(f"Weights for quadratic model for survived dataset are "
      f"{quadratic_weights_0}.")
print(f"Weights for quadratic model for passed dataset are "
      f"{quadratic_weights_1}.")

# Predicting using quadratic model
quadratic_model_0 = np.poly1d(quadratic_weights_0)
quadratic_model_1 = np.poly1d(quadratic_weights_1)
quadratic_predicted_0 = quadratic_model_0(X_test_0)
quadratic_predicted_1 = quadratic_model_1(X_test_1)

quadratic_sse_0 = sse(Y_test_0, quadratic_predicted_0)
quadratic_sse_1 = sse(Y_test_1, quadratic_predicted_1)

"""Plots"""
quad_0 = plt.figure()
lin_plot = quad_0.add_subplot(111)
lin_plot.scatter(X_test_0, Y_test_0, color='black')
lin_plot.plot(X_test_0, quadratic_predicted_0, color='blue', linewidth=2)
quad_0.savefig('quadratic_model_survived')

quad_1 = plt.figure()
lin_plot = quad_1.add_subplot(111)
lin_plot.scatter(X_test_1, Y_test_1, color='black')
lin_plot.plot(X_test_1, quadratic_predicted_1, color='blue', linewidth=2)
quad_1.savefig('quadratic_model_passed')

print(f"Sum of Squared Residuals for surviving dataset and a quadratic linear "
      f"model is {quadratic_sse_0:,.2f}.")
print(f"Sum of Squared Residuals for passed dataset and a quadratic linear "
      f"model is {quadratic_sse_1:,.2f}")

temp_list = ['y=ax^2+bx+c', quadratic_sse_0, quadratic_sse_1]
stats_list.append(temp_list)

# -----------------------------------------------------------------------
"""Cubic"""
print("\n --- Cubic Model ---")
# Splitting data again
X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(df_0_x, df_0_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(df_1_x, df_1_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
# Calculating cubic weights
cubic_weights_0 = np.polyfit(X_train_0, Y_train_0, 3)
cubic_weights_1 = np.polyfit(X_train_1, Y_train_1, 3)

print(f"Weights for cubic model for surviving dataset are {cubic_weights_0}.")
print(f"Weights for cubic model for passed dataset are {cubic_weights_1}.")


cubic_model_0 = np.poly1d(cubic_weights_0)
cubic_model_1 = np.poly1d(cubic_weights_1)

# Predicting values with cubic models
cubic_predicted_0 = cubic_model_0(X_test_0)
cubic_predicted_1 = cubic_model_1(X_test_1)

cubic_sse_0 = sse(Y_test_0, cubic_predicted_0)
cubic_sse_1 = sse(Y_test_1, cubic_predicted_1)

"""Graph here"""

print(f"Sum of Squared Residuals for surviving dataset and a cubic linear "
      f"model is {cubic_sse_0:,.2f}.")
print(f"Sum of Squared Residuals for passed dataset and a cubic linear "
      f"model is {cubic_sse_1:,.2f}")

temp_list = ['y=ax^3+bx^2+cx+d', cubic_sse_0, cubic_sse_1]
stats_list.append(temp_list)

"""y=a log(x) + b"""
print("\n --- y=a log(x) + b Model ---")
# Splitting data again
X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(df_0_x, df_0_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(df_1_x, df_1_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )


glm1_weights_0 = np.polyfit(np.log(X_train_0), Y_train_0, 1)
glm1_weights_1 = np.polyfit(np.log(X_train_1), Y_train_1, 1)

print(f"Weights for model y = a * log(x) + b for the survived dataset are"
      f" {glm1_weights_0}.")
print(f"Weights for model y = a * log(x) + b for the passed dataset are"
      f" {glm1_weights_1}.")

glm1_model_0 = np.poly1d(glm1_weights_0)
glm1_model_1 = np.poly1d(glm1_weights_1)


glm1_predicted_0 = glm1_model_0(np.log(X_test_0))
glm1_predicted_1 = glm1_model_1(np.log(X_test_1))

glm1_sse_0 = sse(Y_test_0, glm1_predicted_0)
glm1_sse_1 = sse(Y_test_1, glm1_predicted_1)

"""Graph here"""

print(f"Sum of Squared Residuals for surviving dataset and generalized linear "
      f"model of y = a * log(x) + b {glm1_sse_0:,.2f}.")
print(f"Sum of Squared Residuals for passed dataset and generalized linear "
      f"model of y = a* log(x) + b is {glm1_sse_1:,.2f}")


temp_list = ['y=alogx+b', glm1_sse_0, glm1_sse_1]
stats_list.append(temp_list)


"""log(y) = a log(x) + b"""
print("\n --- log(y) = a log(x) + b Model ---")
# Splitting data again
X_train_0, X_test_0, Y_train_0, Y_test_0 = train_test_split(df_0_x, df_0_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(df_1_x, df_1_y,
                                                            test_size=0.5,
                                                            random_state=3,
                                                            )


glm2_weights_0 = np.polyfit(np.log(X_train_0), np.log(Y_train_0), 1)
glm2_weights_1 = np.polyfit(np.log(X_train_1), np.log(Y_train_1), 1)

print(f"Weights for model log(y) = a * log(x) + b for the survived dataset are "
      f"{glm2_weights_0}.")
print(f"Weights for mode log(y) = a * log(x) + b for the passed dataset are "
      f"{glm2_weights_1}.")

glm2_model_0 = np.poly1d(glm2_weights_0)
glm2_model_1 = np.poly1d(glm2_weights_1)


glm2_predicted_0 = glm2_model_0(np.log(X_test_0))
glm2_predicted_1 = glm2_model_1(np.log(X_test_1))

glm2_sse_0 = sse(np.log(Y_test_0), glm2_predicted_0)
glm2_sse_1 = sse(np.log(Y_test_1), glm2_predicted_1)

"""Graph here"""

print(f"Sum of Squared Residuals for surviving dataset and generalized linear "
      f"model of log(y) = a * log(x) + b {glm2_sse_0:,.2f}.")
print(f"Sum of Squared Residuals for passed dataset and generalized linear "
      f"model of log(y) = a * log(x) + b is {glm2_sse_1:,.2f}")

temp_list = ['logy=alogx+b', glm2_sse_0, glm2_sse_1]
stats_list.append(temp_list)

"""Plots"""

# Linear plots
linear_0 = plt.figure()
lin_plot = linear_0.add_subplot(111)
plt.title("Linear Model Survived")
lin_plot.scatter(X_test_0, Y_test_0, color='black')
lin_plot.plot(X_test_0, linear_predicted_0, color='blue', linewidth=2)
linear_0.savefig('linear_model_survived')

linear_1 = plt.figure()
lin_plot = linear_1.add_subplot(111)
plt.title("Linear Model Passed")
lin_plot.scatter(X_test_1, Y_test_1, color='black')
lin_plot.plot(X_test_1, linear_predicted_1, color='blue', linewidth=2)
linear_0.savefig('linear_model_passed')


# Quadratic model plots
quad_0 = plt.figure()
quad_plot = quad_0.add_subplot(111)
plt.title('Quad Model Survived')
quad_plot.scatter(X_test_0, Y_test_0, color='black')
quad_plot.plot(X_test_0, quadratic_predicted_0, color='blue', linewidth=2)
quad_0.savefig('quad_model_survived')

quad_1 = plt.figure()
quad_plot = quad_1.add_subplot(111)
plt.title("Quad Model Passed")
quad_plot.scatter(X_test_1, Y_test_1, color='black')
quad_plot.plot(X_test_1, quadratic_predicted_1, color='blue', linewidth=2)
quad_1.savefig('quad_model_passed')

# Cubic model plots
cubic_0 = plt.figure()
cubic_plot = cubic_0.add_subplot(111)
plt.title('Cubic Model Survived')
cubic_plot.scatter(X_test_0, Y_test_0, color='black')
cubic_plot.plot(X_test_0, cubic_predicted_0, color='blue', linewidth=2)
cubic_0.savefig('cubic_model_survived')

cubic_1 = plt.figure()
cubic_plot = cubic_1.add_subplot(111)
plt.title('Cubic Model Passed')
cubic_plot.scatter(X_test_1, Y_test_1, color='black')
cubic_plot.plot(X_test_1, cubic_predicted_1, color='blue', linewidth=2)
cubic_1.savefig('cubic_model_passed')

# y=alogx+b plot
glm1_0 = plt.figure()
glm1_plot = glm1_0.add_subplot(111)
plt.title('y=alogx+b Model Survived')
glm1_plot.scatter(X_test_0, Y_test_0, color='black')
glm1_plot.plot(X_test_0, glm1_predicted_0, color='blue', linewidth=2)
glm1_0.savefig('GLM1_model_survived')

glm1_1 = plt.figure()
glm1_plot = glm1_1.add_subplot(111)
plt.title('y=alogx+b Model Passed')
glm1_plot.scatter(X_test_1, Y_test_1, color='black')
glm1_plot.plot(X_test_1, glm1_predicted_1, color='blue', linewidth=2)
glm1_1.savefig('GLM1_model_passed')

# logy=alogx+b plot
glm2_0 = plt.figure()
glm2_plot = glm2_0.add_subplot(111)
plt.title('logy=alogx+b Model Survived')
glm2_plot.scatter(X_test_0, Y_test_0, color='black')
glm2_plot.plot(X_test_0, glm2_predicted_0, color='blue', linewidth=2)
glm2_0.savefig('GLM2_model_survived')

glm2_1 = plt.figure()
glm2_plot = glm2_1.add_subplot(111)
plt.title('logy=alogx+b Model Passed')
glm2_plot.scatter(X_test_1, Y_test_1, color='black')
glm2_plot.plot(X_test_1, glm2_predicted_1, color='blue', linewidth=2)
glm2_1.savefig('GLM2_model_passed')

# Calling table printer for Question 3
table_printer(stats_list)
