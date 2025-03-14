import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler

from Modeling.feature_engineering import feature_engineering

logger = logging.getLogger(__name__)

def train_models(input_csv: str = "Yelpmerged_data.csv") -> None:
    """
    Reads the CSV, performs feature engineering, then applies:
        1) Univariate SelectKBest
        2) RFE
        3) Random Forest Importance
    to choose features for final model training.

    Finally performs a train/test split, scales the features 
    (only fitting on the training set), 
    and trains Logistic Regression & Random Forest on the final feature set.
    """

    logger.info(f"Starting model training with {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logger.error(f"{input_csv} not found. Ensure data ingestion step was completed.", exc_info=True)
        return

    # 1. Feature Engineering
    logger.info("Applying feature engineering...")
    df = feature_engineering(df)

    # Confirm 'HighRated' exists
    if 'HighRated' not in df.columns:
        logger.error("No 'HighRated' column after feature engineering. Can't train.")
        return

    # 2. Build the numeric feature set, ignoring 'HighRated'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'HighRated' in numeric_cols:
        numeric_cols.remove('HighRated')

    X_numeric = df[numeric_cols].copy()
    y = df['HighRated'].copy()

    # ============== Feature Selection Steps ==============
    logger.info("Performing Univariate Feature Selection (SelectKBest, top=15)...")
    k_best = min(15, X_numeric.shape[1])
    selector_uni = SelectKBest(score_func=f_classif, k=k_best)
    X_univariate = selector_uni.fit_transform(X_numeric, y)

    # Retrieve the chosen feature names
    selected_features_uni = X_numeric.columns[selector_uni.get_support()]
    logger.info("Top features based on Univariate Selection:")
    for i, feature in enumerate(selected_features_uni):
        score_val = selector_uni.scores_[selector_uni.get_support()][i]
        logger.info(f"  {i+1}. {feature} (score: {score_val:.4f})")

    # ============== Pick Final Features ==============
    final_features = list(selected_features_uni)

    # Optionally remove leftover columns that cause perfect scores:
    for col_to_remove in ('stars_business', 'stars_review'):
        if col_to_remove in final_features:
            final_features.remove(col_to_remove)

    logger.info(f"\nFinal chosen features after removing rating columns: {final_features}")

    X_final = X_numeric[final_features].copy()

    # ============== Split THEN Scale =================
    # Train/Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_final, y, test_size=0.3, random_state=42
    )
    logger.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    # Scale only after the split to avoid data leakage
    scaler_global = StandardScaler()
    X_train = scaler_global.fit_transform(X_train_raw)
    X_test = scaler_global.transform(X_test_raw)

    # ============== Models ==============

    # 1) Logistic Regression
    logreg = LogisticRegression(max_iter=10000, class_weight='balanced')
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    lr_cm = confusion_matrix(y_test, y_pred_lr)
    lr_report = classification_report(y_test, y_pred_lr)
    logger.info("Logistic Regression results:")
    logger.info(f"Confusion Matrix:\n{lr_cm}")
    logger.info(f"Classification Report:\n{lr_report}")

    # 2) Random Forest
    rf_balanced = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf_balanced.fit(X_train, y_train)
    y_pred_rf = rf_balanced.predict(X_test)

    rf_cm = confusion_matrix(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf, digits=4)
    logger.info("Random Forest results:")
    logger.info(f"Confusion Matrix:\n{rf_cm}")
    logger.info(f"Classification Report:\n{rf_report}")

    logger.info("Model training completed successfully.")