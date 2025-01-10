import pysubdisc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

# Load the Adult data
data = pd.read_csv('D:/Cloud/OneDrive - Vishwaniketan Trust/Leiden University Workspace/Advances in Data Mining/PA-4/adult.txt')

# Examine input data
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())

print('\n\n******* Section 1 *******\n')

# SECTION 1
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
print(sd.describeSearchParameters())
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 2 *******\n')

# SECTION 2
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.qualityMeasureMinimum = 0.1
sd.numericStrategy = 'NUMERIC_BEST'  # Modified as per feedback
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 3 *******\n')

# SECTION 3
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.refinementDepth = 2
sd.qualityMeasureMinimum = 0.25
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 4 *******\n')

# SECTION 4
sd_no_filter = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd_no_filter.qualityMeasure = 'CORTANA_QUALITY'
sd_no_filter.filterSubgroups = False
sd_no_filter.run()
print(sd_no_filter.asDataFrame())
sd.filterSubgroups = True
print("Subgroup count with filtering turned ON: ", len(sd.asDataFrame()))
print("Subgroup count with filtering turned OFF: ", len(sd_no_filter.asDataFrame()))
top_3_subgroups = sd.asDataFrame().head(3)
print("\nTop 3 Subgroups (Pattern Team of size 3):")
print(top_3_subgroups)

print('\n\n******* Section 5 *******\n')

# SECTION 5
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'RELATIVE_LIFT'
sd.qualityMeasureMinimum = 0.0
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 6 *******\n')

# SECTION 6
sd.minimumCoverage = 5
sd.qualityMeasureMinimum = 3  # Modified as per feedback
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 7 *******\n')

# SECTION 7
sd = pysubdisc.singleNumericTarget(data, 'age')
sd.qualityMeasureMinimum = 0.0
sd.run()
print(sd.asDataFrame())
print("Average age in the dataset:", data['age'].mean())

print('\n\n******* Section 8 *******\n')

# SECTION 8: Compute threshold based on randomization
print("Running 100 random SD runs to compute empirical threshold...")
num_runs = 100
random_qualities = []

for i in range(num_runs):
    data['random_target'] = np.random.permutation(data['target'])
    sd_random = pysubdisc.singleNominalTarget(data, 'random_target', 'gr50K')
    sd_random.run()
    random_df = sd_random.asDataFrame()
    max_quality = random_df['Quality'].max()  # Adjust 'Quality' to whatever the correct name is
    random_qualities.append(max_quality)

threshold = np.percentile(random_qualities, 95)
print(f"Computed empirical threshold: {threshold}")
sd.qualityMeasureMinimum = threshold
sd.computeThreshold(setAsMinimum=True)  # Modified as per feedback
sd.run()
print(sd.asDataFrame())

print('\n\n******* Section 9 *******\n')

# SECTION 9: Perform double regression on Ames Housing dataset
data_housing = pd.read_csv('D:/Cloud/OneDrive - Vishwaniketan Trust/Leiden University Workspace/Advances in Data Mining/PA-4/ameshousing.txt')
table = pysubdisc.loadDataFrame(data_housing)
print(table.describeColumns())
sd = pysubdisc.doubleRegressionTarget(table, 'Lot Area', 'SalePrice')  # Modified as per feedback
X = data_housing['Lot Area'].values
y = data_housing['SalePrice'].values
X_with_intercept = np.vstack([np.ones(len(X)), X]).T
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
intercept, slope = beta
print(f"Linear model: SalePrice = {intercept} + {slope} * Lot Area")
dollars_per_sqft = slope
print(f"Dollars per square foot in the entire dataset: {dollars_per_sqft}")
sd.refinementDepth = 1
sd.run()
print(sd.asDataFrame().loc[0])

print("\nKey Outputs:\n")

print("\nSection 1")
top_subgroups_section_1 = sd.asDataFrame().head()
print(f"Top 5 Subgroups from Section 1:\n{top_subgroups_section_1[['Conditions', 'Quality']]}")

print("\nSection 2")
top_subgroups_section_2 = sd.asDataFrame().head()
print(f"Top 5 Subgroups from Section 2:\n{top_subgroups_section_2[['Conditions', 'Quality']]}")

print("\nSection 3")
top_subgroups_section_3 = sd.asDataFrame().head()
print(f"Top 5 Subgroups from Section 3:\n{top_subgroups_section_3[['Conditions', 'Quality']]}")

print("\nSection 4")
print(f"Subgroup count with filtering ON: {len(sd.asDataFrame())}")
print(f"Subgroup count with filtering OFF: {len(sd_no_filter.asDataFrame())}")
top_subgroups_no_filter = sd_no_filter.asDataFrame().head()
print(f"Top 5 Subgroups with filtering OFF:\n{top_subgroups_no_filter[['Conditions', 'Quality']]}")

print("\nSection 5")
top_subgroups_section_5 = sd.asDataFrame().head()
print(f"Top 5 Subgroups from Section 5 (Relative Lift):\n{top_subgroups_section_5[['Conditions', 'Quality']]}")

print("\nSection 6")
small_subgroups = sd.asDataFrame().head()
print(f"Top 5 Small Subgroups from Section 6:\n{small_subgroups[['Conditions', 'Quality']]}")

print("\nSection 7")
average_age = data['age'].mean()
print(f"Average age in dataset: {average_age}")
best_regression_subgroup = sd.asDataFrame().loc[0]
print(f"Best subgroup from regression on 'age':\n{best_regression_subgroup[['Conditions', 'Quality']]}")

print("\nSection 8")
print(f"Computed empirical threshold: {threshold}")
top_subgroups_section_8 = sd.asDataFrame().head()
print(f"Top 5 Subgroups from Section 8 (after threshold):\n{top_subgroups_section_8[['Conditions', 'Quality']]}")

print("\nSection 9")
print(f"Linear model: SalePrice = {intercept} + {slope} * Lot Area")
print(f"Dollars per square foot in the entire dataset: {dollars_per_sqft}")
best_subgroup_section_9 = sd.asDataFrame().loc[0]
print(f"Best subgroup from Section 9:\n{best_subgroup_section_9[['Conditions', 'Quality']]}")
