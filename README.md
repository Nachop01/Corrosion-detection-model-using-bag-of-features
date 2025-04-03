# Corrosion Detection in Images Using Bag of Features (BoF) with Multiple Feature Descriptors and Classifiers

## Overview
This project implements a corrosion detection system in images using **Bag of Features (BoF)** combined with multiple feature extraction methods and machine learning classifiers. The pipeline evaluates:
- **Feature Descriptors**: HOG, LBP, GLCM, and color descriptors RGB and HSV
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

### 🧠 Classifiers Tested
| Model              | Key Characteristics               |
|--------------------|------------------------------------|
| Random Forest      | Ensemble of decision trees        |
| XGBoost            | Gradient-boosted trees             |
| LightGBM           | High-speed gradient boosting      |
| Logistic Regression| Linear probabilistic classifier    |
| KNN                | Distance-based instance learning  |
| SVM                | Kernel-based separation           |

## Repository Structure
├── feature_extraction/ # Feature extraction scripts

├── Preprocessing/ # Analysis of the dataframe

├── Cross-Validatios and Models / # Training models and 5-fold CV
