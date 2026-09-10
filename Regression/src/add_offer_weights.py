import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "markup_full_real.csv"
OUTPUT_FILE = BASE_DIR / "data" / "weighted_offers.csv"

df = pd.read_csv(INPUT_FILE)

# Count the number of offers in each EDC-month market
df["n_offers"] = (
    df.groupby(["Year", "Month", "EDC"])["EDC"]
      .transform("size")
)

# Give each EDC-month the same total weight
df["weight"] = 1 / df["n_offers"]

# Round only for display/storage
# df["weight"] = df["weight"].round(6)

# Check the total weight of each EDC-month
check = (
    df.groupby(["Year", "Month", "EDC"])["weight"]
      .sum()
      .reset_index(name="total_weight")
)

print("Offer counts and weights:")
print(
    df[
        ["Year", "Month", "EDC", "real_price", "n_offers", "weight"]
    ].head(20)
)

print("\nTotal weight by EDC-month:")
print(check.head(20))

print("\nMinimum total weight:", check["total_weight"].min())
print("Maximum total weight:", check["total_weight"].max())

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)