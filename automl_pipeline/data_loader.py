# Import pandas for data manipulation
import pandas as pd

def load_data(data_path):
    """
    Loads a CSV file and returns features (X) and labels (y) for classification tasks.

    Parameters:
    - data_path (str): Path to the CSV file.

    Returns:
    - X (DataFrame): Feature columns.
    - y (Series): Target labels.
    """

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_path)

    # Check if the expected target column 'label' exists
    if 'label' not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")

    # Separate the features (X) by dropping the label column
    X = df.drop(columns=['label'])

    # Extract the target labels (y)
    y = df['label']

    # Return the features and labels
    return X, y
