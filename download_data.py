import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import requests

def load_german_credit_data():
    """
    Load and preprocess the German Credit Data.
    Returns preprocessed DataFrame with credit risk labels.
    """
    # URLs for the German Credit Data
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    ]

    # Column names for the dataset
    columns = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone',
        'foreign_worker', 'credit_risk'
    ]

    # Check if the data file exists locally
    if os.path.exists('german_credit.csv'):
        print("Loading existing data from german_credit.csv")
        df = pd.read_csv('german_credit.csv')
    else:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Download the raw data file
        raw_data_path = 'data/german.data'
        if not os.path.exists(raw_data_path):
            print("Downloading German Credit Data...")
            success = False

            for url in urls:
                try:
                    response = requests.get(url, verify=False)
                    if response.status_code == 200:
                        with open(raw_data_path, 'wb') as f:
                            f.write(response.content)
                        success = True
                        print(f"Successfully downloaded data from {url}")
                        break
                except Exception as e:
                    print(f"Failed to download from {url}: {str(e)}")
                    continue

            if not success:
                raise Exception("Failed to download the dataset and no local copy available.")

        # Load the data
        print("Processing downloaded data...")
        df = pd.read_csv(raw_data_path, sep=' ', names=columns)

        # Save the processed data
        df.to_csv('german_credit.csv', index=False)
        print("Data saved to german_credit.csv")

    # Convert credit_risk from {1, 2} to {1, 0} where 1 means bad credit
    df['credit_risk'] = df['credit_risk'].map({2: 0, 1: 1})

    # Convert categorical variables to string type for better handling
    categorical_columns = ['status', 'credit_history', 'purpose', 'savings',
                         'employment_duration', 'personal_status_sex', 'other_debtors',
                         'property', 'other_installment_plans', 'housing', 'job',
                         'own_telephone', 'foreign_worker']

    df[categorical_columns] = df[categorical_columns].astype(str)

    return df

if __name__ == "__main__":
    # Load and save the data
    print("Loading German Credit Data...")
    df = load_german_credit_data()
    print("Dataset shape:", df.shape)
    print("\nFeature names:", list(df.columns))
    print("\nSample of the data:")
    print(df.head())
    print("\nCredit risk distribution:")
    print(df['credit_risk'].value_counts(normalize=True))