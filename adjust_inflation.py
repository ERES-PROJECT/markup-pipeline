"""
Inflation adjustment

Formula:
RealPrice_t = NominalPrice_t * (CPI_base / CPI_t)

The latest month in the data is used as the base period.
"""

import pandas as pd

MARKUP_FILE = "markup_full.csv"
CPI_FILE = "cpi_monthly.csv"
OUTPUT_FILE = "markup_full_real.csv"

markup = pd.read_csv(MARKUP_FILE)
cpi = pd.read_csv(CPI_FILE)

markup["Year"] = markup["Year"].astype(int)
markup["Month"] = markup["Month"].astype(int)
cpi["Year"] = cpi["Year"].astype(int)
cpi["Month"] = cpi["Month"].astype(int)

# use the latest month as the base period
periods = (
    markup[["Year", "Month"]]
    .drop_duplicates()
    .sort_values(["Year", "Month"])
)

base_year = int(periods.iloc[-1]["Year"])
base_month = int(periods.iloc[-1]["Month"])

base_row = cpi[
    (cpi["Year"] == base_year) &
    (cpi["Month"] == base_month)
]

if base_row.empty:
    raise ValueError(
        f"CPI not found for {base_year}-{base_month:02d}"
    )

base_cpi = base_row["CPI"].iloc[0]

# match CPI to each month
df = markup.merge(
    cpi,
    on=["Year", "Month"],
    how="left"
)

if df["CPI"].isna().any():
    missing = (
        df.loc[df["CPI"].isna(), ["Year", "Month"]]
        .drop_duplicates()
        .sort_values(["Year", "Month"])
    )

    print("Missing CPI:")
    print(missing.to_string(index=False))
    raise ValueError("Missing CPI values")

# convert prices to base-period dollars
df["inflation_factor"] = base_cpi / df["CPI"]

df["real_price"] = df["price"] * df["inflation_factor"]
df["real_PTC"] = df["PTC"] * df["inflation_factor"]
df["real_markup"] = df["markup"] * df["inflation_factor"]

df["real_price"] = df["real_price"].round(4)
df["real_PTC"] = df["real_PTC"].round(4)
df["real_markup"] = df["real_markup"].round(4)
df["inflation_factor"] = df["inflation_factor"].round(6)

df = df.sort_values(["Year", "Month", "EDC"]).reset_index(drop=True)

df["price"] = df["price"].round(4)
df["PTC"] = df["PTC"].round(4)
df["markup"] = df["markup"].round(4)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Base period: {base_year}-{base_month:02d}")
print(f"Base CPI: {base_cpi}")
print(f"Rows: {len(df)}")
print(f"Saved to {OUTPUT_FILE}")