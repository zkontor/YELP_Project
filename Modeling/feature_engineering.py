import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    #Convert date to YYYY-MM
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).str[:7]

    #Create LongReview from length
    if 'text' in df.columns:
        df['RevLength'] = df['text'].astype(str).apply(len)
        df['LongReview'] = np.where(df['RevLength'] > 150, 1, 0)
        df.drop('RevLength', axis=1, inplace=True)
    else:
        logger.warning("No 'text' column to create LongReview from.")

    #Target Variable: HighRated
    if 'stars_business' in df.columns:
        df['HighRated'] = np.where(df['stars_business'] > 4, 1, 0)
    else:
        logger.warning("No 'stars_business' column to create 'HighRated' target.")

    #Engagement
    if all(col in df.columns for col in ['useful','funny','cool']):
        df['TotalReviewEngagement'] = df[['useful','funny','cool']].sum(axis=1, skipna=True)
        scaler = StandardScaler()
        df['NormEngagement'] = scaler.fit_transform(df[['TotalReviewEngagement']])
    else:
        logger.warning("Missing one of 'useful','funny','cool'. Skipping engagement feature.")

    #Popularity
    if 'review_count' in df.columns:
        avg_review_ct = df['review_count'].mean()
        df['ReviewPopularity'] = np.where(df['review_count'] > avg_review_ct, 1, 0)
    else:
        logger.warning("No 'review_count' column for ReviewPopularity feature.")
   #region or categories
    state_region_mapping = {
        "ME":"Northeast","NH":"Northeast","VT":"Northeast","MA":"Northeast","RI":"Northeast",
        "CT":"Northeast","NY":"Northeast","NJ":"Northeast","PA":"Northeast","OH":"Midwest",
        "MI":"Midwest","IN":"Midwest","IL":"Midwest","WI":"Midwest","MN":"Midwest","IA":"Midwest",
        "MO":"Midwest","ND":"Midwest","SD":"Midwest","NE":"Midwest","KS":"Midwest","DE":"South",
        "MD":"South","WV":"South","VA":"South","KY":"South","TN":"South","NC":"South","SC":"South",
        "GA":"South","FL":"South","AL":"South","MS":"South","AR":"South","LA":"South","OK":"South",
        "TX":"South","DC":"South","MT":"West","ID":"West","WY":"West","NV":"West","UT":"West",
        "CO":"West","AZ":"West","NM":"West","AK":"West","WA":"West","OR":"West","CA":"West","HI":"West",
        "AB":"Other"
    }
    if 'state' in df.columns:
        df['Region'] = df['state'].map(state_region_mapping).fillna("Other")

    #Category
    if 'categories' in df.columns:
        common_categories = ['italian', 'mexican', 'chinese', 'fast food', 'pizza', 'vegan', 'vegetarian']
        for category in common_categories:
            col_name = f"is_{category.replace(' ', '_')}"
            df[col_name] = df['categories'].str.contains(category, case=False, na=False).astype(int)
        df['category_count'] = df['categories'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        logger.info("No 'categories' column to create category-based features.")

    logger.info("Feature engineering complete.")
    return df