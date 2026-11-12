
import pandas as pd
import numpy as np

# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_data/cleaned_CICIDS2017.csv")

# 2. Dataset dimensions
print("DATASET SHAPE:", df.shape)

# 3. First five records
print("\nFIRST 5 ROWS:")
print(df.head().to_string())

# 4. Column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# 5. Class distribution
print("\nCLASS DISTRIBUTION:")
print(df["Class"].value_counts())

print("\nCLASS PERCENTAGES:")
print(df["Class"].value_counts(normalize=True) * 100)

# 6. Missing values
print("\nTOTAL MISSING VALUES:")
print(df.isna().sum().sum())

# 7. Infinite values in numeric columns
numeric = df.select_dtypes(include=np.number)
print("\nTOTAL INFINITE VALUES:")
print(np.isinf(numeric.to_numpy()).sum())

# 8. Duplicates excluding source filename
cols = [c for c in df.columns if c != "Source_File"]
print("\nDUPLICATE ROWS:")
print(df.duplicated(subset=cols).sum())

# 9. Data types
print("\nDATA TYPES:")
print(df.dtypes.value_counts())

# ==========================================
# SECOND INVESTIGATION
# ==========================================

print("\n========== ORIGINAL ATTACK LABELS ==========")

print(df["Label"].value_counts().to_string())

print("\n========== LABELS WITHIN EACH CLASS ==========")

print(
    df.groupby(["Class", "Label"]).size()
      .sort_values(ascending=False).to_string()
)

# Check whether duplicates occur across source files
print("\n========== CROSS-FILE DUPLICATES ==========")

feature_cols = [
    c for c in df.columns if c != "Source_File"
]

# Hash records to identify matching rows efficiently
row_hash = pd.util.hash_pandas_object(
    df[feature_cols], index=False
)

dup_mask = row_hash.duplicated(keep=False)

duplicate_info = pd.DataFrame({
    "hash": row_hash[dup_mask],
    "source": df.loc[dup_mask, "Source_File"].values,
    "class": df.loc[dup_mask, "Class"].values
})

groups = duplicate_info.groupby("hash").agg(
    copies=("hash", "size"),
    source_files=("source", "nunique"),
    classes=("class", "nunique")
)

print("Duplicate groups:", len(groups))
print("Duplicate rows:", int(groups["copies"].sum()))
print("Groups appearing across multiple files:",
      int((groups["source_files"] > 1).sum()))
print("Groups with conflicting classes:",
      int((groups["classes"] > 1).sum()))

# Check potentially duplicated feature columns
print("\n========== FEATURE CHECK ==========")

print(
    "Fwd Header Length equals Fwd Header Length.1:",
    df["Fwd Header Length"].equals(df["Fwd Header Length.1"])
)
