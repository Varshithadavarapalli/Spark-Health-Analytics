# Asthma Disease Prediction Using Apache Spark

## Project Overview
This project aims to leverage the powerful data processing capabilities of Apache Spark to analyze a comprehensive health dataset from Kaggle, focusing on asthma disease prediction. By employing big data technologies, we aim to uncover patterns and insights that can improve diagnostic accuracy and understand the underlying factors contributing to asthma.

## Installation

### Prerequisites
- Apache Spark 3.0+
- Python 3.8+
- Java 8+
- Hadoop 3.2+ (optional for HDFS support)

### Setup
1. **Install Apache Spark**:
   - Download Apache Spark from [the official website](https://spark.apache.org/downloads.html).
   - Unpack the distribution and configure the environment variables:
  
     
2. **Clone the Repository**:
   - Clone this repository to your local machine using:
    
    

3. **Install Python Dependencies**:
   - Install the required Python packages:
     
    

## Usage
To run the analysis scripts, navigate to the project directory and use the following Spark submit command:



code to Asthma Disease Prediction 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy.stats import chi2_contingency

# ===================== 1. Load Dataset ===================== #
file_path = '/home/sat3812/Downloads/asthma_disease_data.csv'  # Update the path as needed
df = pd.read_csv(file_path)
print("Initial Data Sample:\n", df.head())

# ===================== 2. Preprocessing ===================== #
# Step 1: Fill missing values in continuous columns
continuous_columns = ['Age', 'BMI', 'DietQuality', 'LungFunctionFEV1', 'LungFunctionFVC']
df[continuous_columns] = df[continuous_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
print("\nAfter Filling Missing Values:\n", df[continuous_columns].describe())

# Step 2: Encode categorical columns
label_columns = ['Gender', 'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies', 'Eczema', 
                 'HayFever', 'GastroesophagealReflux', 'Wheezing', 'ShortnessOfBreath', 
                 'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced', 'Diagnosis']
label_encoders = {col: LabelEncoder() for col in label_columns}
for col in label_columns:
    df[col] = label_encoders[col].fit_transform(df[col])
print("\nAfter Encoding Categorical Variables:\n", df[label_columns].head())

# Step 3: Scale continuous features
scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
print("\nAfter Scaling Continuous Variables:\n", df[continuous_columns].head())

# Step 4: Binning of continuous variables
binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
df['Age_Binned'] = binner.fit_transform(df[['Age']])
df['BMI_Binned'] = binner.fit_transform(df[['BMI']])
df['LungFunctionFEV1_Binned'] = binner.fit_transform(df[['LungFunctionFEV1']])
print("\nAfter Binning 'Age', 'BMI', and 'LungFunctionFEV1' into Quartiles:")
print(df[['Age', 'Age_Binned', 'BMI', 'BMI_Binned', 'LungFunctionFEV1', 'LungFunctionFEV1_Binned']].head())

# ===================== 3. Split Dataset ===================== #
X = df[[col for col in label_columns if col != 'Diagnosis'] + continuous_columns]
y_class = df['Diagnosis']  # Target for Logistic Regression (classification)
y_reg = df['LungFunctionFEV1']  # Target for Linear Regression (regression)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)

# ===================== 4. Logistic Regression (Classification) ===================== #
logreg = LogisticRegression(max_iter=5000, random_state=42)
logreg.fit(X_train, y_train_class)
train_acc_logreg = accuracy_score(y_train_class, logreg.predict(X_train))
test_acc_logreg = accuracy_score(y_test_class, logreg.predict(X_test))
print(f"\nLogistic Regression Training Accuracy: {train_acc_logreg}")
print(f"Logistic Regression Test Accuracy: {test_acc_logreg}")

# ===================== 5. Linear Regression (Regression) ===================== #
linreg = LinearRegression()
linreg.fit(X_train, y_train_reg)
train_rmse_linreg = np.sqrt(mean_squared_error(y_train_reg, linreg.predict(X_train)))
test_rmse_linreg = np.sqrt(mean_squared_error(y_test_reg, linreg.predict(X_test)))
train_r2_linreg = r2_score(y_train_reg, linreg.predict(X_train))
test_r2_linreg = r2_score(y_test_reg, linreg.predict(X_test))
print(f"\nLinear Regression Training RMSE: {train_rmse_linreg}")
print(f"Linear Regression Test RMSE: {test_rmse_linreg}")
print(f"Linear Regression Training R2 Score: {train_r2_linreg}")
print(f"Linear Regression Test R2 Score: {test_r2_linreg}")

# ===================== 6. Chi-Square Tests for Multiple Categorical Pairs ===================== #
categorical_pairs = [('Gender', 'Diagnosis'), ('PetAllergy', 'Diagnosis'), ('FamilyHistoryAsthma', 'Diagnosis'), 
                     ('HistoryOfAllergies', 'Diagnosis'), ('Eczema', 'Diagnosis')]
for cat1, cat2 in categorical_pairs:
    contingency_table = pd.crosstab(df[cat1], df[cat2])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test between '{cat1}' and '{cat2}': Chi2 = {chi2}, p-value = {p_value}")

# ===================== 7. K-Means Clustering ===================== #
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[continuous_columns])
print("\nK-Means Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels Distribution:\n", df['Cluster'].value_counts())

# ===================== 8. Correlation Analysis ===================== #
correlation_matrix = df[continuous_columns].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# ===================== 9. Correlation Heatmap ===================== #
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


