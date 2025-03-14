import logging
from Logging import configure_global_logger
from DataPrep.DataHandling import merge_yelp_data
from Modeling.train import train_models
from EDA.EDA import run_eda_df
from Modeling.feature_engineering import feature_engineering

def run_pipeline():
   
    configure_global_logger(logging.INFO)
    logging.info("Starting Yelp data pipeline...")

    # Merge data
    business_url = "https://drive.google.com/uc?id=1Ig_oz3clo0yd4Jr06hK8BECcvp6XMcAq"
    review_url   = "https://drive.google.com/uc?id=1216j0qFj33JZVZWjZe9CLqDuht2qBtjZ"

    try:
        merged_df = merge_yelp_data(
            business_file='yelp_academic_dataset_business.json',
            review_file='yelp_academic_dataset_review.json',
            output_file='Yelpmerged_data.csv',
            business_gdrive_url=business_url,
            review_gdrive_url=review_url
        )
        logging.info(f"Merged data shape: {merged_df.shape}")
    except Exception as e:
        logging.error("Error in data ingestion step: %s", e, exc_info=True)
        return
    logging.info("Applying feature engineering in memory for EDA & training.")
    enriched_df = feature_engineering(merged_df)
    
    run_eda_df(enriched_df, output_dir="plots")

    logging.info("Re-saving enriched DF to 'Yelpmerged_data_enriched.csv' for model training.")
    enriched_df.to_csv("Yelpmerged_data_enriched.csv", index=False)

    #train.
    try:
        train_models(input_csv='Yelpmerged_data_enriched.csv')
    except Exception as e:
        logging.error("Error in model training step: %s", e, exc_info=True)
        return

    logging.info("Yelp data pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
