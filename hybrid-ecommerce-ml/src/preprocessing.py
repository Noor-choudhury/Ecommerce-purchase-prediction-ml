import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_COLS = [
    'Administrative', 'Administrative_Duration', 
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

CATEGORICAL_COLS = [
    'Month', 'OperatingSystems', 'Browser', 
    'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

TARGET_COL = 'Revenue'

def find_dataset_path(data_dir="data"):
    """Automatically detects dataset file in directory."""
    if os.path.exists(os.path.join(data_dir, "online_shoppers_intention.csv")):
        return os.path.join(data_dir, "online_shoppers_intention.csv")
    
    # Check parent directory or working directory for fallback
    possible_names = ["UCI Online Shoppers Purchasing Intention.csv", "online_shoppers_intention.csv"]
    for name in possible_names:
        if os.path.exists(name):
            return name
        if os.path.exists(os.path.join(data_dir, name)):
            return os.path.join(data_dir, name)
            
    raise FileNotFoundError("Could not locate Online Shoppers Purchasing Intention CSV file.")

def load_and_clean_data(file_path):
    """Loads dataset and performs basic cleaning without leakage."""
    df = pd.read_csv(file_path)
    # Deduplicate exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def split_data(df, test_size=0.20, random_state=42):
    """Performs stratified train/test split prior to any fitting."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    """Builds ColumnTransformer for strict numerical scaling and categorical encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor