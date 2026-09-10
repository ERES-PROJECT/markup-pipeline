import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "offers_with_rt.csv"
OUTPUT_FILE = BASE_DIR / "data" / "regression_data.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only observations with matched RT data
df = df.dropna(subset=["RT_Average"]).copy()

# Convert nominal RT to real RT using the same inflation factor
df["real_RT_Average"] = (
    df["RT_Average"] * df["inflation_factor"]
).round(4)

df["real_RT_Median"] = (
    df["RT_Median"] * df["inflation_factor"]
).round(4)

print("Rows kept:", len(df))

print("\nSample:")
print(
    df[
        [
            "Year",
            "Month",
            "EDC",
            "real_price",
            "real_PTC",
            "real_markup",
            "RT_Average",
            "real_RT_Average",
            "n_offers",
            "weight"
        ]
    ].head(20)
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)