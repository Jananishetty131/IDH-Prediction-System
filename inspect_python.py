import zipfile
import pandas as pd

file_path = "New_IDH_Dataset/dataset_11.csv.zip"

with zipfile.ZipFile(file_path, "r") as archive:
    with archive.open("dataset_11.csv") as file:
        df = pd.read_csv(
            file,
            sep=";",
            index_col=False,
            usecols=range(25)
        )

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

# Convert the actual target
df["IDH"] = df["D_Hipotension"].map({
    "NO": 0,
    "SI": 1
})

# Remove original target column
df = df.drop(columns=["D_Hipotension"])

print("DATASET SHAPE:", df.shape)

print("\nTARGET DISTRIBUTION:")
print(df["IDH"].value_counts(dropna=False))

print("\nFIRST 5 ROWS:")
print(df.head().to_string(index=False))

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nUNIQUE VALUES IN EACH COLUMN:")

for col in df.columns:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique())
    print("Sample values:", df[col].unique()[:15])