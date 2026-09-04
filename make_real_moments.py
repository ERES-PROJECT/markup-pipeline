import pandas as pd

INPUT_FILE = "markup_full_real.csv"
OUTPUT_FILE = "markup_moments_real.csv"

df = pd.read_csv(INPUT_FILE)

moments = (
    df.groupby(["Year", "Month", "EDC"])
    .agg(
        n_offers=("real_markup", "size"),
        MeanMarkup=("real_markup", "mean"),
        Variance=("real_markup", "var"),
        ShareAbovePTC=("real_markup", lambda x: (x > 0).mean()),
        Q10=("real_markup", lambda x: x.quantile(0.10)),
        Q50=("real_markup", lambda x: x.quantile(0.50)),
        Q90=("real_markup", lambda x: x.quantile(0.90)),
    )
    .reset_index()
)

cols = [
    "MeanMarkup",
    "Variance",
    "ShareAbovePTC",
    "Q10",
    "Q50",
    "Q90",
]

moments[cols] = moments[cols].round(4)

moments = moments.sort_values(
    ["Year", "Month", "EDC"]
).reset_index(drop=True)

moments.to_csv(OUTPUT_FILE, index=False)

print(moments.head())
print()
print(f"Markets: {len(moments)}")
print(f"Saved to {OUTPUT_FILE}")

