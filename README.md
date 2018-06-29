# Restaurant Revenue Predictor



## Data Description
TFI has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants. The data columns include the open date, location, city type, and three categories of obfuscated data: Demographic data, Real estate data, and Commercial data. The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis.

##### Data fields

- **Id** : Restaurant id.
- **Open Date** : opening date for a restaurant
- **City** : City that the restaurant is in. Note that there are unicode in the names.
- **City Group**: Type of the city. Big cities, or Other.
- **Type**: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
- **P1, P2 - P37**: There are three categories of these **obfuscated data.** Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
- **Revenue**: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is **the target of predictive analysis.**


## Workflow:
#### 1. Prepare and EDA
  - Initial intuitions
  - Convert categorical data to dummy variables

#### 2. Split data
   - Hold out
   - Train
   - Test
   - Standardize x's

#### 3. Fit Linear regression - Calculate RMSE
  - On standardized x_train, y_train

#### 4. Try L1 Regularization - Calculate RMSE

#### 5. Manually Engineer Features - Calculate RMSE
  - In depth analysis of assumptions for linear analysis.

#### 6. Choose Best Model




##### To improve:
-  I wouldn't have a hold out set. While, it seems like a great idea, you need to get rid of outliers first! 
