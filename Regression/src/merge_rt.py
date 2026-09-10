import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OFFERS_FILE = BASE_DIR / "data" / "weighted_offers.csv"
RT_FILE = BASE_DIR / "data" / "rt_pa_all_years_summary.csv"
OUTPUT_FILE = BASE_DIR / "data" / "offers_with_rt.csv"

offers = pd.read_csv(OFFERS_FILE)
rt = pd.read_csv(RT_FILE)

# Rename RT columns so they are clear after merging
rt = rt.rename(columns={
    "Average": "RT_Average",
    "Median": "RT_Median"
})

# Merge by Year, Month, and EDC
merged = offers.merge(
    rt,
    on=["Year", "Month", "EDC"],
    how="left"
)

# Check unmatched RT observations
missing_rt = merged["RT_Average"].isna().sum()

print("Total offer rows:", len(merged))
print("Rows without matched RT:", missing_rt)

if missing_rt > 0:
    print("\nUnmatched EDC-months:")
    print(
        merged.loc[
            merged["RT_Average"].isna(),
            ["Year", "Month", "EDC"]
        ]
        .drop_duplicates()
        .sort_values(["Year", "Month", "EDC"])
    )

print("\nSample merged data:")
print(
    merged[
        [
            "Year",
            "Month",
            "EDC",
            "real_price",
            "real_PTC",
            "real_markup",
            "n_offers",
            "weight",
            "RT_Average",
            "RT_Median"
        ]
    ].head(20)
)

merged.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)