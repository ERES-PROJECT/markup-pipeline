import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "rt_pa_all_years_summary.csv"

df = pd.read_csv(DATA_FILE)

# Check duplicate EDC-month observations
duplicates = df.duplicated(
    subset=["Year", "Month", "EDC"],
    keep=False
)

print("Total rows:", len(df))
print("Duplicate EDC-month rows:", duplicates.sum())

if duplicates.any():
    print("\nDuplicate observations:")
    print(
        df.loc[duplicates]
        .sort_values(["Year", "Month", "EDC"])
    )

# Count months available for each EDC
coverage = (
    df.groupby("EDC")
      .size()
      .reset_index(name="n_months")
)

print("\nNumber of months by EDC:")
print(coverage)

# Check number of EDCs in each month
market_counts = (
    df.groupby(["Year", "Month"])
      .size()
      .reset_index(name="n_edcs")
)

print("\nEDCs available in each month:")
print(market_counts.head(20))