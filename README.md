# Life Expectancy Survey - ML Regression Analysis

## Project Overview

This project focuses on predicting life expectancy using machine learning regression techniques. The analysis is based on a comprehensive dataset containing various socio-economic and health indicators from different countries over multiple years.

## Dataset Description

The dataset contains **2,938 entries** with **22 features** including:

### Key Features:
- **Target Variable**: Life expectancy
- **Demographics**: Country, Year, Status (Developed/Developing)
- **Health Indicators**: Adult Mortality, Infant deaths, Under-five deaths, HIV/AIDS, Hepatitis B, Measles, Polio, Diphtheria
- **Socio-economic Factors**: GDP, Population, Income composition of resources, Schooling
- **Health Infrastructure**: BMI, Alcohol consumption, Total expenditure, Percentage expenditure
- **Nutritional Factors**: Thinness (1-19 years), Thinness (5-9 years)

## Project Structure

### 1. Exploratory Data Analysis (`life_expectancy_eda.ipynb`)

This notebook contains comprehensive exploratory data analysis including:

#### Data Quality Assessment:
- **Missing Values Analysis**: Identified and handled missing values using median imputation
- **Duplicate Detection**: Confirmed no duplicate rows in the dataset
- **Outlier Detection**: Used IQR method and 3-sigma rule to identify outliers across all numerical features

#### Statistical Analysis:
- **Descriptive Statistics**: Mean, median, standard deviation, range, variance
- **Distribution Analysis**: Skewness and kurtosis calculations
- **Mode Analysis**: For categorical variables

#### Data Visualization:
- **Univariate Analysis**: Histograms, box plots for all numerical features
- **Bivariate Analysis**: Bar charts for categorical variables vs life expectancy
- **Multivariate Analysis**: Correlation heatmap showing relationships between all features
- **Outlier Visualization**: Box plots before and after outlier treatment

#### Key Findings:
- Life expectancy ranges from 36.3 to 89.0 years
- Strong correlations found between life expectancy and factors like:
  - Adult Mortality (negative correlation)
  - Income composition of resources (positive correlation)
  - Schooling (positive correlation)
  - HIV/AIDS (negative correlation)

### 2. Machine Learning Model (`life_expectancy_ml.ipynb`)

This notebook implements various regression models to predict life expectancy:

#### Data Preprocessing:
- **Missing Value Treatment**: Median imputation for numerical features
- **Categorical Encoding**: Label encoding for Country and Status variables
- **Feature Selection**: Selected 8 most relevant features based on correlation analysis
- **Data Normalization**: MinMax scaling for feature normalization
- **Train-Test Split**: 80-20 split with random state for reproducibility

#### Selected Features:
1. HIV/AIDS
2. Income composition of resources
3. Adult Mortality
4. Under-five deaths
5. Schooling
6. BMI
7. Thinness (1-19 years)
8. Status

#### Models Implemented:

1. **Linear Regression**
   - Training Score: 0.786
   - R² Score: 0.788
   - MAE: 3.026
   - RMSE: 4.171

2. **Support Vector Regression (SVR)**
   - Training Score: 0.909
   - R² Score: 0.911
   - MAE: 1.867
   - RMSE: 2.700

3. **Random Forest Regressor**
   - Training Score: 0.967
   - R² Score: 0.955
   - MAE: 1.266
   - RMSE: 1.913

4. **Gradient Boosting Regressor**
   - Training Score: 0.992
   - R² Score: 0.974
   - MAE: 0.849
   - RMSE: 1.474

#### Model Optimization:
- **Hyperparameter Tuning**: Used RandomizedSearchCV for Random Forest optimization
- **Winsorization**: Applied 5% winsorization to handle extreme outliers
- **Ensemble Methods**: Implemented Voting Regressor combining multiple models

#### Final Results (Post-Optimization):
- **Best Performing Model**: Gradient Boosting Regressor
- **R² Score**: 0.977
- **MAE**: 0.732
- **RMSE**: 1.287

## Key Insights

1. **Health Factors**: Adult mortality and HIV/AIDS rates are strong predictors of life expectancy
2. **Socio-economic Impact**: Income composition and schooling significantly influence life expectancy
3. **Nutritional Factors**: BMI and thinness indicators show moderate correlation with life expectancy
4. **Model Performance**: Ensemble methods and gradient boosting provide the best predictive accuracy

## Technical Requirements

### Python Libraries Used:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `scipy` - Statistical functions

### Installation:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage

1. **EDA Analysis**: Run `life_expectancy_eda.ipynb` to explore the dataset and understand data patterns
2. **Model Training**: Execute `life_expectancy_ml.ipynb` to train and evaluate regression models
3. **Results**: Compare model performance metrics and select the best performing model

## Future Enhancements

- Implement deep learning models (Neural Networks)
- Add more feature engineering techniques
- Implement time series analysis for temporal patterns
- Create a web application for interactive predictions
- Add cross-validation for more robust model evaluation

## Contributing

Feel free to contribute to this project by:
- Adding new features or models
- Improving data preprocessing techniques
- Enhancing visualization capabilities
- Optimizing model performance

## License

This project is open source and available under the MIT License.

---

**Note**: This analysis was conducted as part of a machine learning regression study focusing on life expectancy prediction using various socio-economic and health indicators.
