import os
import json
import logging
import pandas as pd
import gdown

logger = logging.getLogger(__name__)

def download_if_missing(gdrive_url: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        logger.info(f"{local_path} not found locally. Attempting download from Google Drive...")
        gdown.download(gdrive_url, local_path, quiet=False)
        if os.path.exists(local_path):
            logger.info(f"Successfully downloaded {local_path}")
        else:
            logger.error(f"Failed to download from {gdrive_url}")
            raise FileNotFoundError(f"Could not download {local_path}")

def merge_yelp_data(
    business_file: str = 'yelp_academic_dataset_business.json',
    review_file: str = 'yelp_academic_dataset_review.json',
    output_file: str = 'Yelpmerged_data.csv',
    business_gdrive_url: str = "https://drive.google.com/file/d/1Ig_oz3clo0yd4Jr06hK8BECcvp6XMcAq/view?usp=drive_link",
    review_gdrive_url: str = "https://drive.google.com/file/d/1216j0qFj33JZVZWjZe9CLqDuht2qBtjZ/view?usp=drive_link"
) -> pd.DataFrame:
    logger.info("Starting to merge Yelp data...")

#Download business file if missing
    if not os.path.exists(business_file):
        if business_gdrive_url:
            download_if_missing(business_gdrive_url, business_file)
        else:
            logger.error(f"{business_file} not found and no GDrive URL provided.")
            raise FileNotFoundError(f"{business_file} not found. Please provide or download manually.")

    # Download review file if missing
    if not os.path.exists(review_file):
        if review_gdrive_url:
            download_if_missing(review_gdrive_url, review_file)
        else:
            logger.error(f"{review_file} not found and no GDrive URL provided.")
            raise FileNotFoundError(f"{review_file} not found. Please provide or download manually.")

    #Read business data
    logger.debug(f"Reading business data from {business_file}...")
    with open(business_file, 'r', encoding='utf-8') as bf:
        business_data = [json.loads(line) for line in bf]
    business_df = pd.json_normalize(business_data)
    logger.debug(f"business_df shape: {business_df.shape}")

    # Read review data
    logger.debug(f"Reading review data from {review_file}...")
    with open(review_file, 'r', encoding='utf-8') as rf:
        review_data = [json.loads(line) for line in rf]
    review_df = pd.json_normalize(review_data)
    logger.debug(f"review_df shape: {review_df.shape}")

    #relevant columns
    business_columns = [
        'business_id', 'name', 'address', 'city', 'state', 'postal_code',
        'latitude', 'longitude', 'stars', 'review_count', 'categories'
    ]
    review_columns = [
        'review_id', 'user_id', 'business_id', 'stars', 'date', 'text',
        'useful', 'funny', 'cool'
    ]
    business_df = business_df[business_columns]
    review_df = review_df[review_columns]

    # Merge the DataFrames
    merged_df = pd.merge(
        review_df,
        business_df,
        on='business_id',
        how='inner',
        suffixes=('_review', '_business')
    )
    logger.debug(f"merged_df shape after merge: {merged_df.shape}")
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Merged data saved to '{output_file}'")

    return merged_df