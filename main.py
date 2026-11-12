
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("cleaned_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found in the data folder.")

print("Files found:", len(files))


# --------------------------------------------------
# 2. Read and combine all files
# Preserve source filename for later analysis
# --------------------------------------------------

df_list = []

for file in files:
    print("Reading:", file.name)

    temp = pd.read_csv(file, low_memory=False)

    temp.columns = temp.columns.str.strip()

    if "Label" not in temp.columns:
        raise ValueError(f"Label column missing in {file.name}")

    temp["Source_File"] = file.name

    df_list.append(temp)

combined_df = pd.concat(df_list, ignore_index=True)

print("\nOriginal combined shape:", combined_df.shape)


# --------------------------------------------------
# 3. Clean labels
# --------------------------------------------------

combined_df["Label"] = (
    combined_df["Label"]
    .astype("string")
    .str.replace("\ufffd", "-", regex=False)
    .str.strip()
)

print("\nOriginal label distribution:")
print(combined_df["Label"].value_counts(dropna=False))


# --------------------------------------------------
# 4. Remove duplicates within each source file
# Do not remove cross-file duplicates yet
# --------------------------------------------------

original_columns = [
    col for col in combined_df.columns
    if col != "Source_File"
]

before = len(combined_df)

cleaned_parts = []

for source, group in combined_df.groupby(
    "Source_File", sort=False
):
    group = group.drop_duplicates(
        subset=original_columns
    )
    cleaned_parts.append(group)

combined_df = pd.concat(
    cleaned_parts, ignore_index=True
)

print(
    "\nDuplicates removed within source files:",
    before - len(combined_df)
)


# --------------------------------------------------
# 5. Handle infinity and missing values
# --------------------------------------------------

numeric_columns = combined_df.select_dtypes(
    include=np.number
).columns

combined_df[numeric_columns] = (
    combined_df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

before = len(combined_df)

required_columns = [
    col for col in combined_df.columns
    if col != "Source_File"
]

combined_df = combined_df.dropna(
    subset=required_columns
).copy()

print(
    "Rows removed due to NaN/Infinity:",
    before - len(combined_df)
)


# --------------------------------------------------
# 6. Create three classification groups
# --------------------------------------------------

dos_ddos_labels = [
    "DoS Hulk",
    "DDoS",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest"
]

def regroup_label(label):

    if label == "BENIGN":
        return "Normal"

    elif label in dos_ddos_labels:
        return "DoS_DDoS"

    else:
        return "Other_Attacks"


combined_df["Class"] = (
    combined_df["Label"].apply(regroup_label)
)


# --------------------------------------------------
# 7. Final dataset statistics
# --------------------------------------------------

print("\nFinal cleaned shape:", combined_df.shape)

print("\nOriginal attack labels:")
print(combined_df["Label"].value_counts())

print("\nThree-class distribution:")
print(combined_df["Class"].value_counts())

print("\nClass percentages:")
print(
    combined_df["Class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\nRecords by source file:")
print(combined_df["Source_File"].value_counts())


# --------------------------------------------------
# 8. Check cross-file duplicate count
# --------------------------------------------------

cross_file_duplicates = combined_df.duplicated(
    subset=original_columns,
    keep=False
).sum()

print(
    "\nRows involved in possible cross-file duplicates:",
    cross_file_duplicates
)


# --------------------------------------------------
# 9. Save final cleaned dataset
# --------------------------------------------------

output_file = OUTPUT_DIR / "cleaned_CICIDS2017.csv"

combined_df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Final rows:", len(combined_df))
print("Final columns:", len(combined_df.columns))
