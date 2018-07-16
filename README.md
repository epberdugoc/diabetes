# diabetes

Here we include a description of each file in this repository.

clean_data.py - 
Refines and extracts the necessary features based on the paper "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records". Input: diabetic_data.csv. Output: diabetic_data_cleaned.csv

dt_roc.py - 
Saves an ROC plot for the Decision Tree model.

tree_utils.py -
Utility functions for building and constructing trees.

output_utils.py -
Utility functions for visualizing the trees we create.

diabetic_data.csv - 
Raw data file from the UCI Machine Learning Repository of all samples and all features.

diabetic_data_cleaned.csv - 
Data file after being cleaned by clean_data.py.  Output of clean_data.py.

IDs_mapping.csv - 
Table of ID numbers and their meaning.  ID numbers correspond to various entries in diabetic_data.csv.
