from sklearn.ensemble import RandomForestClassifier

def get_prediction_model(random_state=42):
    """Instantiates the core Random Forest classifier for conversion prediction."""
    return RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=random_state
    )