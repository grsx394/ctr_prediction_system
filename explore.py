import pandas as pd

# Loading only first 10,000 rows
df = pd.read_csv('C:/Users/hp/OneDrive/Desktop/CTR_project/data/train.gz', nrows=10000)

print("Shape:", df.shape)  #rows and columns
print("\nColumns:", df.columns.tolist())  # Lists all column names
print("\nData types:\n", df.dtypes)  # Shows whether each column is numeric or text
print("\nClick distribution:\n", df['click'].value_counts()) #Shows how many rows have click=0 vs click=1
print("\nMissing values:\n", df.isnull().sum())  #Shows how many missing values exist per column
print("\nFirst 3 rows:\n", df.head())
