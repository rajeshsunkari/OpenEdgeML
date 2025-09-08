# Import necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_select_model(X_train, X_test, y_train, y_test):
    """
    Trains multiple classification models and selects the best one based on accuracy.

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels

    Returns:
    - best_model: The model with the highest accuracy on the test set
    - best_model_name: Name of the best performing model
    - best_accuracy: Accuracy score of the best model
    """

    # Dictionary of models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Initialize variables to track the best model
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # Iterate through each model, train it, and evaluate its accuracy
    for name, model in models.items():
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test data
        predictions = model.predict(X_test)

        # Calculate accuracy of the model
        accuracy = accuracy_score(y_test, predictions)

        # Print the accuracy for this model
        print(f"{name} Accuracy: {accuracy:.4f}")

        # Update the best model if this one has higher accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Print the selected best model
    print(f"Selected Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # Return the best model and its details
    return best_model, best_model_name, best_accuracy
