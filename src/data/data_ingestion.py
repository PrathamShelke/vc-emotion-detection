import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

# logging configure

#1)Creating Logger
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#2)Creating StreamHandler(To print messages in Console)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#3)Creating FileHandler(To print messages in File and Logging it)
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

#4)Initialiasing Format of messages to be Logged using Formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#5)Adding Both the Handlers in Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except KeyError as e:
        logger.error("Key Error in Params File")
    except yaml.YAMLError as e:
        logger.error("YAML Error")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.ParserError as e:
        raise Exception(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the data: {e}")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        raise Exception(f"Key error: {e} not found in DataFrame.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the data: {e}")

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        raise Exception(f"An error occurred while saving the data: {e}")

def main():
    try:
        test_size = load_params("params.yaml")
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Making New directory data,raw to store train.csv, test.csv files
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
