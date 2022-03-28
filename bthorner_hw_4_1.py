"""
Brian Horner
CS 677 - Summer 2
Date: 8/3/2021
Week 4 Homework Question 1
This program creates correlation matrix plots on specified features and
saves them.
"""

# Imports
import pandas as pd
import seaborn as sns

# Fixing font size to fit feature names
sns.set(font_scale=.5)

"""Question 1.1"""
# Loading csv into dataframe
heart_df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# Cleaning up column names
heart_df.columns = ['Age', 'Anaemia', 'CPK', 'Diabetes',
                    'Ejection_fraction', 'High_blood_pressure',
                    'Platelets', 'Serum Creatinine', 'Serum Sodium',
                    'Sex', 'Smoking', 'Time', 'Death Event'
                    ]


# Splitting dataframe based off death event boolean
df_0 = heart_df.loc[heart_df['Death Event'] == 0]
df_1 = heart_df.loc[heart_df['Death Event'] == 1]

# Grabbing specific columns from death event dataframes
df_1 = df_1.loc[:, ['CPK', 'Serum Creatinine', 'Serum Sodium',
                    'Platelets']]

df_0 = df_0.loc[:, ['CPK', 'Serum Creatinine', 'Serum Sodium',
                    'Platelets']]


"""Question 1.2"""
# Getting Correlation Matrix
M_0 = df_0.corr()
M_1 = df_1.corr()

# Creating Correlation Matrix Plots
survived_plot = sns.heatmap(M_0, annot=True)
passed_plot = sns.heatmap(M_1, annot=True)

# Adding titles to heat maps
survived_plot.set_title("Survived Heat Map")
passed_plot.set_title("Passed Heat Map")

# Grabbing figures from heatmap objects
survived_plot = survived_plot.get_figure()
passed_plot = passed_plot.get_figure()

# Saving plots
passed_plot.savefig('passed_corr.pdf')
survived_plot.savefig('survived_corr.pdf')


"""Question 1.3"""
print("--- Question 1.3 ---\n")
print("Question 1.3 (a) - which features have the highest correlation for "
      "surviving patients?")
print("Creatinine and Serum Sodium have the highest correlation for surviving "
      "patients.\n")


print("Question 1.3 (b) - which features have the lowest correlation for "
      "surviving patients")
print("Serum Sodium and Serum Creatinine have the lowest correlation "
      "for surviving patients.\n")


print("Question 1.3 (c) - which features have the highest correlation for "
      "deceased patients?")
print("Creatinine and Serum Sodium have the highest correlation for deceased "
      "patients.\n")


print("Question 1.3 (d) - which features have the lowest correlation for "
      "deceased patients?")
print("Serum Sodium and Serum Creatinine have the lowest correlation "
      "for deceased patients.\n")


print("Question 1.3 (e) - are the results the same for both cases?")
print("Yes the highest and lowest correlation features are the same for both "
      "cases.\n")
