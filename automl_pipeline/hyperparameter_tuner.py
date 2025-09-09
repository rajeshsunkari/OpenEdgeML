from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def tune_random_forest(X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter tuning using GridSearchCV for RandomForestClassifier.

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels

    Returns:
    - best_model: The best RandomForestClassifier model found
    - best_params: The best hyperparameters
    - best_accuracy: Accuracy of the best model on the test set
    """

    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit the model on training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate accuracy on test data
    predictions = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, predictions)

    return best_model, best_params, best_accuracy

