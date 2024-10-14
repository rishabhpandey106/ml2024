import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

chunksize = 50000  # adjust based on memory capacity
n_estimators_per_chunk = 50  # number of trees per chunk
max_chunks = 8  # since my dataset is 400,000 rows and i want to process all chunks

chunk_iter = pd.read_csv('data_new.csv', chunksize=chunksize)

estimators = []

# Keep a separate test set for final evaluation
test_data = pd.read_csv('data_new.csv', nrows=50000, skiprows=range(1, 400000))
X_test_final = test_data.iloc[:, 1:].values
y_test_final = test_data.iloc[:, 0].values

for i, chunk in enumerate(chunk_iter):
    print(f"Training on chunk {i + 1}")
    
    X = chunk.iloc[:, 1:].values
    y = chunk.iloc[:, 0].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    rf = RandomForestClassifier(n_estimators=n_estimators_per_chunk, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    joblib.dump(rf, f'rf_model_chunk_{i + 1}.pkl')
    
    estimators.extend(rf.estimators_)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Chunk {i + 1} accuracy: {accuracy}")

    if i + 1 >= max_chunks:  # i want to stop after processing 8 chunks (400,000 / 50,000 = 8)
        break

# final random forest classifier with all estimators
final_rf = RandomForestClassifier(n_estimators=len(estimators), random_state=1)

final_rf.fit(X_test_final, y_test_final) 
final_rf.estimators_ = estimators

joblib.dump(final_rf, 'final_random_forest_model.pkl')

y_pred_final = final_rf.predict(X_test_final)
final_accuracy = accuracy_score(y_test_final, y_pred_final)
print(f"Final model accuracy on test set: {final_accuracy}")
