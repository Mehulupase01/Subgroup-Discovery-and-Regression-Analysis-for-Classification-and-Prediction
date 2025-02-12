import pysubdisc
import pandas
import matplotlib.pyplot as plt

# ---------------------------
# Load and describe the dataset
# ---------------------------
data = pandas.read_csv('adult.txt')
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())

# ---------------------------
# Section 1: Basic Nominal Target Discovery
# ---------------------------
print('\n\n******* Section 1 *******\n')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
print(sd.describeSearchParameters())
sd.run()
print(sd.asDataFrame())

# ---------------------------
# Section 2: Custom Quality Measure Settings
# ---------------------------
print('\n\n******* Section 2 *******\n')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.qualityMeasureMinimum = 0.1
sd.searchDepth = 1
sd.numericStrategy = 'NUMERIC_BEST'
sd.run(verbose=False)
print(sd.asDataFrame())

# ---------------------------
# Section 3: Increased Depth and Higher Quality Threshold
# ---------------------------
print('\n\n******* Section 3 *******\n')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.qualityMeasureMinimum = 0.25
sd.searchDepth = 2
sd.numericStrategy = 'NUMERIC_BEST'
sd.run(verbose=False)
print(sd.asDataFrame())

# ---------------------------
# Section 4: Discovery Without Filtering & Pattern Team Extraction
# ---------------------------
print('\n\n******* Section 4 *******\n')
sd_no_filter = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd_no_filter.qualityMeasure = 'CORTANA_QUALITY'
sd_no_filter.qualityMeasureMinimum = 0.25
sd_no_filter.searchDepth = 2
sd_no_filter.numericStrategy = 'NUMERIC_BEST'
sd_no_filter.filterSubgroups = False
sd_no_filter.run(verbose=False)
print(sd_no_filter.asDataFrame())

print("Subgroup count with filtering turned ON: ", len(sd.asDataFrame()))
print("Subgroup count with filtering turned OFF: ", len(sd_no_filter.asDataFrame()))

patternteam_size = 3
patternteam = sd_no_filter.getPatternTeam(patternteam_size)
print("\nPattern Team (of Size 3): \n", patternteam)

# Reset filtering options if needed later
sd_no_filter.filterSubgroups = True
sd.filterSubgroups = True

# ---------------------------
# Section 5: Nominal Target with RELATIVE_LIFT
# ---------------------------
print('\n\n******* Section 5 *******\n')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'RELATIVE_LIFT'
sd.qualityMeasureMinimum = 0.0
sd.searchDepth = 2
sd.numericStrategy = 'NUMERIC_BEST'
sd.run(verbose=False)
print(sd.asDataFrame())

# ---------------------------
# Section 6: RELATIVE_LIFT with Minimum Coverage
# ---------------------------
print('\n\n******* Section 6 *******\n')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'RELATIVE_LIFT'
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasureMinimum = 3
sd.minimumCoverage = 5
sd.run(verbose=False)
print(sd.asDataFrame())

# ---------------------------
# Section 7: Numeric Target ('age') with Z_SCORE
# ---------------------------
print('\n\n******* Section 7 *******\n')
sd = pysubdisc.singleNumericTarget(data, 'age')
sd.qualityMeasure = 'Z_SCORE'
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.qualityMeasureMinimum = 0.0
sd.minimumCoverage = int(0.1 * len(data))
sd.run(verbose=False)
print("Average age in the data: ", data['age'].mean())
print(sd.asDataFrame())

# ---------------------------
# Section 8: Compute Threshold for Significance
# ---------------------------
print('\n\n******* Section 8 *******\n')
sd = pysubdisc.singleNumericTarget(data, 'age')
sd.qualityMeasure = 'Z_SCORE'
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 2
sd.minimumCoverage = int(0.1 * len(data))
sd.computeThreshold(significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100, setAsMinimum=True)
sd.run(verbose=False)
print("Minimum quality for significance: ", sd.qualityMeasureMinimum)
print(sd.asDataFrame())

# ---------------------------
# Section 9: Double Regression Target on Housing Data
# ---------------------------
print('\n\n******* Section 9 *******\n')
data = pandas.read_csv('ameshousing.txt')
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())
sd = pysubdisc.doubleRegressionTarget(data, 'Lot Area', 'SalePrice')
sd.qualityMeasure = 'REGRESSION_SSD_COMPLEMENT'
sd.numericStrategy = 'NUMERIC_BEST'
sd.searchDepth = 1
sd.run(verbose=False)
print(sd.asDataFrame().loc[0])
