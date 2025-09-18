# Optimal Transport in Predictive Mean Matching

The baseline MICE algorithm uses Predictive Mean Matching (PMM) to impute missing data. For each target feature, it fits a coditional model (in our case OLS) on the remaining features to generate predictions for observed rows (donors) and missing rows (recipients). For each recipient, it collects the k nearest donors (by a 1-D gap) and copies one donor’s observed value at random.

This donor criterion collapses the covariate space into a single scalar, effectively throwing away the distribution of the covariates. In datasets with a high ratio of features to instances, this problem can worsen. Optimal Transport Predictive Mean Matching (OT-PMM) tries to address this by replacing the matching rule with a 2 part cost that blends the prediction gap with a covariate distance term, encouraging matches that are close in both prediction and covariates. The relative influence of each term is controllable by alpha and/or beta.

# 

For the data, we used 3 datasets with varying ratios of features to instances from the UCI Machine Learning repo:
- Breast Cancer Wisconsin (Diagnostic) - 30 features and 596 instances
- Blood Transfusion Service Centre - 4 features and 748 instances
- Parkinsons - 22 features and 197 instances

To assess the predictive power and accuracy of PMM and OT-PMM, we used both MAE and RMSE.

To simulate missing data, we defined MCAR, MAR and MNAR (quantile) functions and created 30 masks for each dataset (aiming at around 30% missing data)

#

 The following data shows the percentage improvement of OT-PMM over PMM for the ratio of means and standard deviation:
 
 <img width="699" height="600" alt="MCAR_plot" src="https://github.com/user-attachments/assets/2d0a8550-262e-45c7-8139-f6c998d518c8" />

 <img width="699" height="600" alt="MAR_plot" src="https://github.com/user-attachments/assets/45ba9831-aaa8-408b-aaa5-25cf23644085" />

 <img width="699" height="600" alt="MNAR_quant_plot" src="https://github.com/user-attachments/assets/cf5a06a8-e5bc-4fe2-a031-6d718db803c6" />

For MCAR and MAR, we see small gains in accuracy with mixed stability. If the predictors don't carry useful local structure, the extra distance term in OT-PMM just adds noise without any significant upside.

For MNAR, accuracy improves significantly with little to no loss in stability, especially on the breast cancer and parkinsons datasets. OT-PMM performs poorly under all 3 mechanisms for the transfusion dataset, consistent with our hypothesis that the covariate terms offers limited value in low feature settings. 

OT-PMM seems to provide substantial gains in performance over PMM on datasets with a higher number of features, especially under MNAR which is the most realistic and trickiest missing data to impute.

# 

Notes:
Optimal Transport matching is model agnositc. It's not limited to MICE runs that use OLS for the conditional model; tree based learners and other regressors can also in theory benefit from OT in the matching stage.
