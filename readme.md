Yelp Data Pipeline

Description: This repository merges and processes a large Yelp dataset, performs Exploratory Data Analysis (EDA), and trains models to predict whether a restaurant is “HighRated.” The pipeline is split into multiple modules for clarity:

Data Ingestion: Download/merge raw Yelp JSON data from Google Drive into a single CSV.
Feature Engineering: Create additional features (e.g., sentiment scores, region mapping).
EDA: Performs static EDA (plots saved to disk) 
Model Training: Train logistic regression and random forest models to classify “HighRated” restaurants.




License
This project is licensed under the MIT License.
