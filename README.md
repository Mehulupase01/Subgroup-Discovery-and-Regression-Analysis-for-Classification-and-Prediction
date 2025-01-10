# Subgroup Discovery and Regression Analysis for Classification and Prediction
This project uses the pySubDisc package for Subgroup Discovery (SD) and Regression tasks. It applies Subgroup Discovery to the Adult dataset for classification, and explores the relationship between Lot Area and SalePrice using double regression with the Ames Housing dataset.

# Subgroup Discovery and Regression Analysis for Classification and Prediction

This project applies **Subgroup Discovery (SD)** and **Regression** techniques to analyze datasets using the **pySubDisc** package. It includes subgroup discovery tasks on the **Adult dataset** for classification and a regression analysis on the **Ames Housing dataset** to examine the relationship between lot area and sale price.

## Overview

The project focuses on the following tasks:

1. **Subgroup Discovery (SD)**: Using the **Adult dataset**, discover subgroups based on the target value `gr50K` (whether a person earns more than $50k annually). Different parameters are explored, such as refinement depth, quality measures, and numeric strategies.
2. **Regression Analysis**: Analyze the relationship between **Lot Area** and **SalePrice** using **double regression** on the **Ames Housing dataset**. This analysis explores linear models and subgroup discovery using regression as the target.

### Key Tasks:
1. **Section 1**: Subgroup discovery on the Adult dataset with default parameters.
2. **Section 2**: Apply a different numeric strategy (`best`) and observe differences in subgroup quality.
3. **Section 3**: Adjust refinement depth to 2 and set a higher minimum quality threshold.
4. **Section 4**: Investigate the effect of filtering subgroups on the results.
5. **Section 5**: Use **Relative Lift** as the quality measure for subgroup discovery.
6. **Section 6**: Identify smaller subgroups with a high-quality threshold.
7. **Section 7**: Use **Regression** with the target `age` and analyze subgroup quality.
8. **Section 8**: Compute an empirical threshold based on randomization of the target.
9. **Section 9**: Apply **double regression** on the Ames Housing dataset to analyze **Lot Area** and **SalePrice**.

## Code Structure

The project consists of a Python file (`PA4_4017633.py`) that includes the implementation for each of the above sections. The pySubDisc package is used for performing subgroup discovery and regression tasks.

### Libraries Used:
- **pysubdisc**: For subgroup discovery and regression.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **matplotlib**: For plotting (although plots are skipped in the code).

### Example Command:

```bash
python Subgroup Discovery and Regression Analysis.py
```

## Output

The program prints out the results of subgroup discovery, including subgroups' conditions and quality scores. The results for each section are displayed, with specific focus on the subgroups' coverage, conditions, and quality measures.

For example, the output for Section 1 shows:

```
Top 5 Subgroups from Section 1:
Conditions                  Quality
age <= 23.0                 -0.19
age <= 28.0                 -0.33
age <= 32.0                 -0.35

```

## Results & Discussion

- **Section 1**: Subgroup discovery with default parameters reveals subgroups based on attributes like age, education, and hours-per-week, with varying quality scores.
- **Section 2**: Using the `best` numeric strategy yields different subgroups, improving quality scores.
- **Section 3**: Increasing refinement depth and raising the quality threshold leads to fewer but more significant subgroups.
- **Section 4**: Disabling subgroup filtering shows more subgroups, but many are spurious and irrelevant.
- **Section 5**: Using **Relative Lift** as a quality measure identifies subgroups that perform well based on a different metric.
- **Section 6**: Finding smaller subgroups with a high-quality threshold identifies more targeted patterns.
- **Section 7**: Regression on the **age** attribute identifies interesting subgroups related to age.
- **Section 8**: Computing the empirical threshold helps define a significance level for the quality measure.
- **Section 9**: The double regression on the **Ames Housing dataset** reveals a linear relationship between **Lot Area** and **SalePrice**.

## Conclusion

This project successfully demonstrates **Subgroup Discovery (SD)** techniques for classification and regression tasks. By applying various quality measures and numeric strategies, meaningful subgroups were discovered in the **Adult dataset**. The regression analysis on the **Ames Housing dataset** highlights important insights into the relationship between **Lot Area** and **SalePrice**.

## References

1. **SubDisc GitHub Repository**: https://github.com/SubDisc/SubDisc
2. **pySubDisc GitHub Repository**: https://github.com/SubDisc/pySubDisc
3. **The Adult Dataset**: UCI Machine Learning Repository.
4. **Ames Housing Dataset**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
