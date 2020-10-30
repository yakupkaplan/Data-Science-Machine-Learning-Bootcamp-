# STATISTICS FOR DATA SCIENCE

### Sample Theory

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns

# Define a population
population = np.random.randint(0, 80, 10000)
population[0:10]
population.mean()

# Define a sample from the population that we defined
np.random.seed(115)
sample = np.random.choice(a=population, size=100)
sample[0:10]
sample.mean()

# Define 10 different samples from the population that we defined
np.random.seed(10)
sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)
sample9 = np.random.choice(a=population, size=100)
sample10 = np.random.choice(a=population, size=100)

# Show the mean of these samples
(sample1.mean() + sample2.mean() + sample3.mean() + sample4.mean() + sample5.mean()
 + sample6.mean() + sample7.mean() + sample8.mean() + sample9.mean() + sample10.mean()) / 10

# Show the mean for some samples
sample1.mean()
sample7.mean()
sample8.mean()


### Confidence Interval

prices = np.random.randint(10, 90, 100)
prices.mean() # 46.19
sms.DescrStatsW(prices).tconfint_mean() # (41.626109739464134, 50.75389026053586)


### Probability Distributions

## Bernoulli Distribution

from scipy.stats import bernoulli
p = 0.6
rv = bernoulli(p)
rv.pmf(k=0)
rv.pmf(k=1)

## Binom Distribution

from scipy.stats import binom
p = 0.01
n = 100
rv = binom(n, p)
rv.pmf(1)
rv.pmf(5)
rv.pmf(10)

## Poisson Distribution

from scipy.stats import poisson
lambda_ = 0.1
rv = poisson(mu=lambda_)
rv.pmf(0)
rv.pmf(3)
rv.pmf(5)

## Normal Distribution

from scipy.stats import norm

# P(X < 90)
norm.cdf(90, 80, 5)

# P(X > 90)
1 - norm.cdf(90, 80, 5)
# P(X > 70)
1 - norm.cdf(70, 80, 5)
# P(90 > X > 85)
norm.cdf(90, 80, 5) - norm.cdf(85, 80, 5)


### Law of Large Numbers

# Defines the long-term stability of probability distributions.
# As the number of experiments increases, the difference between the observed value and the expected value increases.
rng = np.random.RandomState(123)
rng.randint(0, 2, size=6)

rng = np.random.RandomState(123)

for i in np.arange(1, 21):
    experiment_count = 2 ** i
    heads_or_tails = rng.randint(0, 2, size=experiment_count)
    tails_probability = np.mean(heads_or_tails)
    print("Flip Count:", experiment_count, "---", 'Probability for Tails: %.2f' % (tails_probability * 100))


### HYPOTHESIS TESTS

import scipy.stats as stats

## One Sample T Test

measurements = np.array([17, 160, 234, 149, 145, 107, 197, 75, 201, 225, 211, 119,
                     157, 145, 127, 244, 163, 114, 145, 65, 112, 185, 202, 146,
                     203, 224, 203, 114, 188, 156, 187, 154, 177, 95, 165, 50, 110,
                     216, 138, 151, 166, 135, 155, 84, 251, 173, 131, 207, 121, 120])

measurements[0:10]

stats.describe(measurements)

## Assumption Test --> Normality Assumption

# Histogram
pd.DataFrame(measurements).plot.hist()
plt.show()

# qqplot
import pylab
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()

## Shapiro-Wilks Test

# H0: Örnek dağılımı ile teorik normal dağılım arasında ist.ol.anl.bir fark.yoktur
# H1: ...fark vardır

from scipy.stats import shapiro
shapiro(measurements)
print("Normallik Testi İçin T Hesap İstatistiği: " + str(shapiro(measurements)[0]))
print("Hesaplanan P-value: " + str(shapiro(measurements)[1]))
0.78 < 0.05 # H0 can not be rejected.

# p - value < ise 0.05 'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.


## Implementation of Hypothesis

# H0: Web sitemizde geçirilen ortalama süre 170 'tir
# H1: ..değildir

stats.ttest_1samp(measurements, popmean=170)

0.0344 < 0.05 # H0 can be rejected.

# p - value < ise 0.05 'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.

measurements.mean() # 154.38 So, we can say, that the mean is less than 170.


## Nonparametric One Sample Test --> If the normality assumption is not satisfied.

from statsmodels.stats.descriptivestats import sign_test

sign_test(measurements, 170)
0.06 < 0.05 # If the normality assumption were not satisfied, we could  reject H0.

# p - value < ise 0.05 'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.

# One Sample Porportion Test

from statsmodels.stats.proportion import proportions_ztest

# HO: p = 0.125
# H1: p != 0.125

count = 40
nobs = 500
value = 0.125

proportions_ztest(count, nobs, value)
0.0002 < 0.05

40 / 500


### Independent Two Samples T Test --> AB Testing

'''
Steps to follow:
    - Define hypothesises and interpretation
    - Check data, outliers, assumptions (Normality and Variance Homogenity) 
    - Implement appropriateness test (parametric/nonparametric, dependent/independent) according to scenario and assumptions
    - Check p-value and interpret the results
'''


# Define a function to show the result for a Hypothesis Test
def hypothesis_test_result_old(p_value):
    if p_value < 0.05:
        print('p-value: %.4f, so H0 can be rejected!' % p_value)
    else:
        print('p-value: %.4f, so H0 can NOT be rejected!' % p_value)


# Define a function to show the result for a Hypothesis Test
def hypothesis_test_result(test_result):
    test_statistics, p_value = test_result
    if p_value < 0.05:
        print('Test Statistics = %.4f, p-value = %.4f, so that H0 can be rejected!' % (test_statistics, p_value))
    else:
        print('Test Statistics = %.4f, p-value = %.4f, so that H0 can NOT be rejected!' % (test_statistics, p_value))

# Show the values separately in two columns

A = pd.DataFrame([30, 27, 21, 27, 29, 30, 20, 20, 27, 32, 35, 22, 24, 23, 25, 27, 23, 27, 23,
                  25, 21, 18, 24, 26, 33, 26, 27, 28, 19, 25])

B = pd.DataFrame([37, 39, 31, 31, 34, 38, 30, 36, 29, 28, 38, 28, 37, 37, 30, 32, 31, 31, 27,
                  32, 33, 33, 33, 31, 32, 33, 26, 32, 33, 29])

A_B = pd.concat([A, B], axis=1)
A_B.columns = ["A", "B"]

A_B.head()

# Show the values in one column and show the class name in another column

A = pd.DataFrame([30, 27, 21, 27, 29, 30, 20, 20, 27, 32, 35, 22, 24, 23, 25, 27, 23, 27, 23,
                  25, 21, 18, 24, 26, 33, 26, 27, 28, 19, 25])

B = pd.DataFrame([37, 39, 31, 31, 34, 38, 30, 36, 29, 28, 38, 28, 37, 37, 30, 32, 31, 31, 27,
                  32, 33, 33, 33, 31, 32, 33, 26, 32, 33, 29])

# A and Group A
GROUP_A = np.arange(len(A))
GROUP_A = pd.DataFrame(GROUP_A)
GROUP_A[:] = "A"
A = pd.concat([A, GROUP_A], axis=1)

# B and Group B
GROUP_B = np.arange(len(B))
GROUP_B = pd.DataFrame(GROUP_B)
GROUP_B[:] = "B"
B = pd.concat([B, GROUP_B], axis=1)

# All the data
AB = pd.concat([A, B])
AB.columns = ["Revenue", "GROUP"]
print(AB.head())
print(AB.tail())

# Compare two dataframes
A_B.head()
AB.head()
AB["GROUP"].value_counts()
AB.groupby('GROUP').agg({'Revenue': np.mean})
# Visualization
sns.boxplot(x="GROUP", y="Revenue", data=AB);
plt.show()


## Assumption Control

# 1.Normality Assumption
# 2.Variance Homogenity

# Normality Assumption

from scipy.stats import shapiro

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

shapiro(A_B.A) # pvalue=0.7962
shapiro(A_B.B) # pvalue=0.2458

hypothesis_test_result(shapiro(A_B.A)) # p-value: 0.7963, so H0 can NOT be rejected! --> Normality OK.
hypothesis_test_result(shapiro(A_B.B)) # p-value: 0.2458, so H0 can NOT be rejected! --> Normality OK.

# p - value < ise 0.05'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.


# Variance Homogenity Assumption

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

stats.levene(A_B.A, A_B.B) # pvalue=0.2964
hypothesis_test_result(stats.levene(A_B.A, A_B.B)) # p-value: 0.2964, so H0 can NOT be rejected! --> Variance Homogenity OK.

# p - value < ise 0.05'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.

## Look for p-value and interpret the results
# Implementation of Hypothesis

# H0: M1 = M2(Eski sistem ile yeni sistem söz konusu olduğunda iki grup ortalamları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2(... vardır)

stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True) # # p-value: 0.0000
test_statistics, pvalue = stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)
print('Test Statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue)) # Test Statistics = -7.0287, p-value = 0.0000

hypothesis_test_result(stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)) # p-value: 0.0000, so H0 can be rejected! --> Statistically, there is a difference between two groups!
# Test Statistics = -7.0287, p-value = 0.0000, so that H0 can be rejected! Statiscticlly, THERE IS A DIFFERENCE! --> The second (NEW) group is better!

# p - value < ise 0.05 'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDILEMEZ.


## Nonparametric Independent Two Samples Test --> If normality assumption is not satisfied! (Mannwhitneyu Test)

# If Normality assumption is satisfied, but variance homogeneity is not --> Then write equal_var=False. That implements Welch Test, automatically!

stats.mannwhitneyu(A_B["A"], A_B["B"]) # p-value = 0.0000
test_statistics, pvalue = stats.mannwhitneyu(A_B["A"], A_B["B"])
print('Test Statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue)) # Test Statistics = 89.5000, p-value = 0.0000

hypothesis_test_result(stats.mannwhitneyu(A_B["A"], A_B["B"])) # Test Statistics = 89.5000, p-value = 0.0000, so that H0 can be rejected!


## Dependent Two Samples T Test

before = pd.DataFrame([123, 119, 119, 116, 123, 123, 121, 120, 117, 118, 121, 121, 123, 119,
                       121, 118, 124, 121, 125, 115, 115, 119, 118, 121, 117, 117, 120, 120,
                       121, 117, 118, 117, 123, 118, 124, 121, 115, 118, 125, 115])

after = pd.DataFrame([118, 127, 122, 132, 129, 123, 129, 132, 128, 130, 128, 138, 140, 130,
                        134, 134, 124, 140, 134, 129, 129, 138, 134, 124, 122, 126, 133, 127,
                        130, 130, 130, 132, 117, 130, 125, 129, 133, 120, 127, 123])

## First Dataset
SEPARATE = pd.concat([before, after], axis=1)
SEPARATE.columns = ["BEFORE", "AFTER"]
print("'SEPARATE' Dataset: \n\n ", SEPARATE.head(), "\n\n")

## Second Dataset
# Creating BEFORE FLAG/TAG
GROUP_BEFORE = np.arange(len(before))
GROUP_BEFORE = pd.DataFrame(GROUP_BEFORE)
GROUP_BEFORE[:] = "ONCESI"
# Concatenate FLAG and BEFORE
A = pd.concat([before, GROUP_BEFORE], axis=1)
# Creating AFTER FLAG/TAG
GROUP_AFTER= np.arange(len(after))
GROUP_AFTER = pd.DataFrame(GROUP_AFTER)
GROUP_AFTER[:] = "AFTER"
# Concatenate FLAG and AFTER
B = pd.concat([after, GROUP_AFTER], axis=1)
# Concatenate all the data
TOGETHER = pd.concat([A, B])
TOGETHER

## NAMING
TOGETHER.columns = ["PERFORMANCE", "BEFORE_AFTER"]
print("'TOGETHER' Dataset: \n\n", TOGETHER.head(), "\n")

## Visualization
import seaborn as sns
sns.boxplot(x="BEFORE_AFTER", y="PERFORMANCE", data=TOGETHER);
plt.show()

## Assumption Controls

## Normality Assumption

from scipy.stats import shapiro

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

shapiro(SEPARATE.BEFORE) # pvalue=0.1072
hypothesis_test_result(shapiro(SEPARATE.BEFORE)) # Test Statistics = 0.9544, p-value = 0.1072, so that H0 can NOT be rejected!

shapiro(SEPARATE.AFTER) # pvalue=0.6159
hypothesis_test_result(shapiro(SEPARATE.AFTER)) # Test Statistics = 0.9780, p-value = 0.6160, so that H0 can NOT be rejected!

# Variance Homogenity Assumption

import scipy.stats as stats

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

stats.levene(SEPARATE.BEFORE, SEPARATE.AFTER) # pvalue=0.0050
hypothesis_test_result(stats.levene(SEPARATE.BEFORE, SEPARATE.AFTER)) # Test Statistics = 8.3130, p-value = 0.0051, so that H0 can be rejected!


## Hypothesis Test

stats.ttest_rel(SEPARATE.BEFORE, SEPARATE.AFTER) # p-value = 0.00000
test_statistics, pvalue = stats.ttest_rel(SEPARATE["BEFORE"], SEPARATE["AFTER"])
print('Test Statistics= %.5f, p-value = %.5f' % (test_statistics, pvalue)) # Test Statistics= -9.28153, p-value = 0.00000

hypothesis_test_result(stats.ttest_rel(SEPARATE.BEFORE, SEPARATE.AFTER)) # Test Statistics = -9.2815, p-value = 0.0000, so that H0 can be rejected!


## Nonparametric Dependent Two Samples --> Wilcoxon Test

stats.wilcoxon(SEPARATE.BEFORE, SEPARATE.AFTER) # p-value = 0.0000
test_statistics, pvalue = stats.wilcoxon(SEPARATE["BEFORE"], SEPARATE["AFTER"])
print('Test Statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue)) # Test Statistics = 15.0000, p-value = 0.0000


## Two Samples Proportion Test --> Conversion Rate Problems

# Asssumption: n1, n2 > 30

from statsmodels.stats.proportion import proportions_ztest

# H0: P1 = P2 --> Yeşil butonunun dönüşüm oranı ile kırmızı butonun dönüşüm oranı arasında ist.ol.anlamlı farklılık yoktur.
# H1: P1 != P2 --> ... vardır

success_count = np.array([300, 250])
observation_count = np.array([1000, 1100])

proportions_ztest(count=success_count, nobs=observation_count) # pvalue=0.0002
hypothesis_test_result(proportions_ztest(count=success_count, nobs=observation_count)) # Test Statistics = 3.7858, p-value = 0.0002, so that H0 can be rejected!
# THERE IS A DIFFERENCE between BUTTONS!


### AB TESTING

## PRICING

# A game company gave gift coins to its users for purchasing items in a game.
# Using these virtual coins, users buy various vehicles for their characters.
# The game company did not specify a price for an item and provided users to buy this item at the price they wanted.
# For example, for the item named shield, users will buy this shield by paying the amounts they see fit.
# For example, a user can pay with 30 units of virtual money given to him, while the other user can pay with 45 units.
# Therefore, users can buy this item with the amounts they can afford to pay.

# Problems to be solved:
# 1. Does the item's price differ by category? Express it statistically.
# 2. What should the price of the item be depending on the first question? Explain why?
# 3. It is desirable to be "mobile" about the price. Create a decision support system for the price strategy and
# 4. Simulate item purchases and income for possible price changes.

df = pd.read_csv("datasets/pricing.csv", sep=";")
df.head()

df["price"].describe()
df["category_id"].value_counts()
df.groupby("category_id").agg({"price": [np.mean, np.median]})


