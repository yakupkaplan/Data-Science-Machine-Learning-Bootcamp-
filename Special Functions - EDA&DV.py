# EDA AND DATA VISUALIZATION SPECIAL FUNCTIONS

'''
A wonderful guide to exploratory data analysis and data visualization for every dataset.

    __Steps for EDA__:

    - GENERAL / GENERAL OVERVIEW / GENERAL PICTURE
    - shape, info(), columns, index, describe(), isnull().any, isnull().sum()

    - CATEGORICAL VARIABLE ANALYSIS
        - Catching categorical variables
        - Number of categorical variables
        - Classes
        - Frequencies
        - Basic visualization --> barplot, countplot

    - NUMERICAL VARIABLE ANALYSIS
        - Catching numerical variables
        - describe with quantiles to see whether there are extraordinary values or not
        - Basic visualization --> histogram, boxplot and density

    - TARGET ANALYSIS
        - Target analysis according to categorical variables --> target_summary_with_cats()
        - Target analysis according to numerical variables --> target_summary_with_nums()
    
    - ANALYSIS OF NUMERCIAL VARIABLES IN COMPARISON WITH EACH OTHER
        - scatterplot
        - lmplot
        - correlation
'''

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv("C:/Users/yakup/OneDrive/Masaüstü/DSMLBC/datasets/train.csv")
df.head()

# Basic visualization rules
# categorical variables --> barplot, countplot
# numerical variables --> histogram, boxplot and density

## Categorical Variables

# Let's catch categorical variables
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
cat_cols


# Define a function that catches categorical variables, prints the distribution and ratio of unique values and finally creates a countplot
def cats_summary1(data):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10]
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(),
                            "Ratio": 100 * data[col].value_counts() / len(data)}), end="\n\n\n")
        sns.countplot(x=col, data=data)
        plt.show()


cats_summary1(df)

# Let's ignore secret cat_cols for now
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
cat_cols


# Define a function takes dataframe, categorical columns and if required number of classes (default value = 10).
# Then, it prints the distribution and ratio of unique values for the variables, which have less than 10 (number of classes) unique values.
# Afterwards, it reports categorical variables it described, how many numerical variables, but seem categorical we have and finally report these variables.
def cats_summary2(data, categorical_cols, number_of_classes=10):
    var_count = 0  # reporting how many categorical variables are there?
    vars_more_classes = []  # save the variables that have classes more than a number that we determined

    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # select according to number of classes
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary2(df, cat_cols)

# See the distribuiton of variables according to quantiles
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T


## NUMERICAL VARIABLES

# Number of Numerical variables
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols

num_cols = [col for col in df.columns if df[col].dtypes != "O"
            and col not in ["PassengerId", "Survived"]]
num_cols


# Define a function that plots histogram of numerical variables and prints the number of numerical variables if plotted.
def hist_for_nums(data, numeric_cols):
    col_counter = 0

    for col in numeric_cols:
        data[col].hist()
        plt.xlabel(col)
        plt.title(col)
        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

## TARGET ANALYSIS

# Target Analysis according to categorical variables

df["Survived"].value_counts()

df.groupby("Sex")["Survived"].mean()

cat_cols = [col for col in df.columns if len(df[col].unique()) < 10]

print('Number of Categorical Variables: ', len(cat_cols))
print(cat_cols)


# Define a function that catches categorical variables with class number less than 10 and exclude target variable and analyse target variable with respect to categories
def target_summary_with_cats(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]

    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")


target_summary_with_cats(df, "Survived")

# Target Analysis according to numerical variables

df.groupby("Survived").agg({"Age": np.mean})


# Define a function that catches numerical variables exclude target variable ang groupby survived with respect to numerical variables
def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "PassengerId"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")


target_summary_with_nums(df, "Survived")

## Analysis Of Numerical Variables In Comparison With Each Other

df = sns.load_dataset("tips")
df.head()

sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()

sns.lmplot(x="total_bill", y="tip", data=df)
plt.show()

df.corr()

df = pd.read_csv("C:/Users/yakup/OneDrive/Masaüstü/DSMLBC/datasets/train.csv")
df.head()


# Special function to summarize this notebook

def analyse(data, target):
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in target]

    print(
        '********************************Let\'s see categorical variables and take a quick look at them!********************************\n')
    cats_summary1(data)
    print(
        '********************************Let\'s see numerical variables and take a quick look at them with histograms!********************************\n')
    hist_for_nums(data, num_cols)
    print(
        '********************************Let\'s see target summary with categorical variables!********************************\n')
    target_summary_with_cats(data, target)
    print(
        '********************************Let\'s see target summary with numerical variables!********************************\n')
    target_summary_with_nums(data, target)


analyse(df, "Survived")

## Catching NAN columns

nan_columns = df.columns[df.isna().any()].tolist()
nan_columns


######### RECAP ###########

def cat_summary1(data):
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10]
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(),
                            "Ratio": 100 * data[col].value_counts() / len(data)}), end="\n\n\n")
        sns.countplot(x=col, data=data)
        plt.show()


def cat_summary2(data, categorical_cols, number_of_classes=10):
    var_count = 0  # reporting how many categorical variables are there?
    vars_more_classes = []  # save the variables that have classes more than a number that we determined

    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # select according to number of classes
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}), end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


def hist_for_nums(data, numeric_cols):
    col_counter = 0

    for col in numeric_cols:
        data[col].hist()
        plt.xlabel(col)
        plt.title(col)
        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted")


def target_summary_with_cats(data, target):
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target]

    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")


def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "PassengerId"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")
