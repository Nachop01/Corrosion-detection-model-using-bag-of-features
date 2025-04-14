# Corrosion Detection in Images Using Bag of Features (BoF) with Multiple Feature Descriptors and Classifiers

## Overview
This project implements a corrosion detection system in images using the Bag of Features (BoF) model. It combines multiple feature extraction methods with machine learning classifiers to deliver accurate results. The pipeline is designed for applications such as industrial inspection or infrastructure maintenance.

## Key Highlights

- **Feature Descriptors**: HOG, LBP, GLCM, and colour descriptors RGB and HSV
- **Classifiers**: Random Forest, XGBoost, LightGBM, Logistic Regression, KNN, and SVM  
- **Feature Selection**: Performance comparison with/without wrapper-based variable selection (Sequential Forward Selector)

Designed for industrial inspection or infrastructure maintenance applications.

## Key Features
### 🔍 Feature Extraction
1. **HOG (Histogram of Oriented Gradients)**  
   Captures edge orientations and gradient magnitudes.
2. **LBP (Local Binary Patterns)**  
   Encodes texture information via local pixel comparisons.
3. **GLCM (Gray-Level Co-occurrence Matrix)**  
   Quantifies texture through spatial pixel relationships.
4. **RGB Color Descriptors**  
   Statistical measures (mean, std, skewness, energy, entropy) per channel.
5. **HSV Color Descriptors**  
   Statistical measures (mean, std, skewness, energy, entropy) per channel.

### ⚙️ Methodology
1. **Bag of Features Pipeline**  
   - Obtain image patches with different shapes
   - Extract local features from image patches
   - Obtain the label of the mask patches
2. **Wrapper Method**  
   Sequential Forward Selector (SFS) for optimal feature subset selection.
3. **Cross-Validation**  
   Stratified 5-fold CV to assess model robustness.

## Data Source

The images used in this project were obtained from the following dataset:

Bianchi, E., & Hebdon, M. (2021). *Corrosion Condition State Semantic Segmentation Dataset* (Version 2) [Dataset]. University Libraries, Virginia Tech. [https://doi.org/10.7294/16624663.v2](https://doi.org/10.7294/16624663.v2)

This dataset provides annotated images for corrosion condition state semantic segmentation and is designed to support research in infrastructure maintenance and inspection.


### 🧠 **Classifiers Tested**

| **Model**           | **Key Characteristics**            | **Hyperparameters**                                                                 | **Grid**                                                                                   | **Code**                                                                 |
|----------------------|-------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Random Forest**    | Ensemble of decision trees         | - Number of trees<br>- Split criterion<br>- Min samples per leaf                    | (25, 50, 75, 100)<br>(Entropy, Gini Index)<br>(1, 2, 4, 8, 16, 32)                        | [View Code](./src/ML_models/Random_Forest.ipnyb)                                     |
| **XGBoost**          | Gradient-boosted trees            | - Number of iterations<br>- Learning rate<br>- Min samples per leaf                | (25, 50, 75, 100, 125, 150)<br>(0.025, 0.05, 0.1, 0.2, 0.4)<br>(1, 2, 4, 8, 16, 32)      | [View Code](./src/ML_models/XGBOOST.ipnyb)                                           |
| **LightGBM (GOSS)**  | High-speed gradient boosting       | - High gradient rate<br>- Low gradient rate<br>- Learning rate<br>- Number of leaves| (0.2, 0.4, 0.6)<br>(0.05, 0.1, 0.3)<br>(0.025, 0.05, 0.1, 0.2, 0.4)<br>(10, 30, 50)      | [View Code](./src/ML_models/LIGHTGBM_GOSS.ipnyb)                                     |
| **LightGBM (GBDT)**  | High-speed gradient boosting       | - Number of iterations<br>- Learning rate<br>- Number of leaves                    | (25, 50, 75, 100, 125, 150)<br>(0.01, 0.05, 0.1, 0.5)<br>(10, 30, 50)                    | [View Code](./src/ML_models/LIGHTGBM_GBDT.ipnyb)                                     |
| **Logistic Regression** | Linear probabilistic classifier   | - Coefficients ($\beta_1,\dots,\beta_d$)                                           | $\beta_i \in \mathbb{R}$                                                                  | [View Code](./src/ML_models/Logistic_regression.ipnyb)                               |
| **SVM**              | Kernel-based separation            | - Kernel<br>- Error tolerance (C)                                                  | (Linear, Gaussian)<br>(0.001, 0.01, 0.1, 1)                                              | [View Code](./src/ML_models/SVM.ipnyb)                                               |
| **KNN**              | Distance-based instance learning   | - Distance metric<br>- Number of neighbors (k)                                     | (Euclidean, Manhattan)<br>(1, 3, 5, 11, 21, 41, 61)                                      | [View Code](./src/ML_models/KNN.ipnyb)                                               |


## Repository Structure
├── feature_extraction/ # Feature extraction scripts

├── Preprocessing/ # Analysis of the dataframe

├── Cross-Validatios and Models / # Training models and 5-fold CV
