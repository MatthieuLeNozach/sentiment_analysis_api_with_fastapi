from celery import shared_task

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

@shared_task
def run_regression():
    # Generate synthetic dataset
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
    
    # Convert the first feature to integer, second to float, and third to categorical
    X[:, 0] = X[:, 0].astype(int)
    X[:, 1] = X[:, 1].astype(float)
    X[:, 2] = np.random.choice(['a', 'b', 'c', 'd'], size=100)
    
    # Define the preprocessing for the categorical feature
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), [2])
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing and regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions.tolist()