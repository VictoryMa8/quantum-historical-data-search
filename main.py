import pandas as pd

# Converting from parquet to CSV format
parquet_file_path = "data.parquet"
csv_file_path = "data.csv"

# Read the Parquet file into a pandas dataframe
dataframe = pd.read_parquet("hf://datasets/burgerbee/history_wiki/data/train-00000-of-00001.parquet")

# Write the dataframe into a CSV file
# Setting index to False to prevent writing the dataframe index as a column in the CSV
dataframe.to_csv(csv_file_path, index=False)