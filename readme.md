Yelp Data Pipeline

Description: This repository merges and processes a large Yelp dataset, performs Exploratory Data Analysis (EDA), and trains models to predict whether a restaurant is “HighRated.” The pipeline is split into multiple modules for clarity:

Data Ingestion: Download/merge raw Yelp JSON data from Google Drive into a single CSV.
Feature Engineering: Create additional features (e.g., sentiment scores, region mapping).
EDA: Performs static EDA (plots saved to disk) 
Model Training: Train logistic regression and random forest models to classify “HighRated” restaurants.



Project Structure
my_yelp_project/
├── DataPrep/
│   ├── DataHandling.py        # Merges & downloads raw data
│   └── __init__.py
├── Modeling/
│   ├── feature_engineering.py # Adds sentiment, region, etc.
│   ├── train.py               # Feature selection & model training
│   └── __init__.py
├── EDA/
│   ├── EDA.py                 # Static EDA with matplotlib/seaborn
│   ├── newEDA.py (optional)   # Interactive EDA with Dash
│   └── __init__.py
├── Logging.py                 # Global logger config
├── main.py                    # Orchestrates entire pipeline
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── .gitignore



License
This project is licensed under the MIT License.
