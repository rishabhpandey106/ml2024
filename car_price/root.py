import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import make_column_transformer
from scipy.sparse import csr_matrix, hstack

# Load data
data = pd.read_csv('car_price/model/cleaned_data.csv')
data = data.drop(data.columns[0], axis=1)

# Check for missing values
if data.isnull().values.any():
    data.fillna(data.mean(), inplace=True)  # Fill NaN with mean or handle as appropriate

# Split into features and target
X = data.drop(columns='Price')
y = data['Price']

# scaler = StandardScaler()
# X[['year', 'kms_driven']] = scaler.fit_transform(X[['year', 'kms_driven']])

# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X[['year', 'kms_driven']])
# poly_feature_names = poly.get_feature_names_out(input_features=['year', 'kms_driven'])
# X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# X = pd.concat([X.drop(columns=['year', 'kms_driven']), X_poly_df], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)

# Custom linear regression class (from scratch)
class LinearRegressionScratch:
    def __init__(self,alpha=0.0):
        self.intercept_ = None
        self.coef_ = None
        self.alpha = alpha
    
    def fit(self, X, y):
        print(f"Shape of X before adding intercept: {X.shape}")
        ones = csr_matrix(np.ones((X.shape[0], 1)))  # Create a CSR matrix of ones
        X_b = hstack([ones, X])  # Concatenate the column of ones with X
        print(f"Shape of X after adding intercept: {X_b.shape}")

        # Convert to dense array
        X_b_dense = X_b.toarray()
        identity = np.eye(X_b.shape[1])
        # Calculate coefficients using the normal equation
        b = np.linalg.inv(X_b_dense.T @ X_b_dense + self.alpha * identity) @ X_b_dense.T @ y.values
        
        self.intercept_ = b[0]
        self.coef_ = b[1:]
    
    def predict(self, X):
        return self.intercept_ + np.dot(X, self.coef_)

# One-hot encoding for categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

# Define column transformer for encoding
column_transformer = make_column_transformer(
    (ohe, ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Custom Pipeline Structure
class CustomPipeline:
    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def fit(self, X, y):
        # Transform the data
        X_transformed = self.transformer.fit_transform(X)
        print(f"Shape of transformed X: {X_transformed.shape}")
        if X_transformed.size == 0:
            raise ValueError("Transformed data has 0 size. Check the input features.")
        
        # Fit the custom model
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        # Transform the test data
        X_transformed = self.transformer.transform(X)
        return self.model.predict(X_transformed)

# Initialize custom pipeline with column transformer and linear regression model
pipe = CustomPipeline(transformer=column_transformer, model=LinearRegressionScratch(alpha=1.0))

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predict on test data
input_data = pd.DataFrame([{
    'name': 'Audi Q7',
    'company': 'Audi',
    'year': 2014,
    'kms_driven': 16934,
    'fuel_type': 'Diesel'
}])
y_pred = pipe.predict(input_data)
print(y_pred)
# Evaluate the model
# print(f"R2 score (custom model): {r2_score(y_test, y_pred)}")
