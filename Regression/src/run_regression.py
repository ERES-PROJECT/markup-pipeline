import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "regression_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "regression_markup_results.txt"

df = pd.read_csv(DATA_FILE)

# Keep observations with the variables needed for the regression
df = df.dropna(subset=["real_markup", "real_RT_Average", "Year", "EDC"])

print("Number of observations:", len(df))

# Regression:
# real markup = RT wholesale price + year fixed effects + EDC fixed effects
model = smf.wls(
    formula="real_markup ~ real_RT_Average + C(Year) + C(EDC)",
    data=df,
    weights=df["weight"]
).fit(cov_type="HC1")

print(model.summary())

# Save results
with open(OUTPUT_FILE, "w") as f:
    f.write(model.summary().as_text())

print("\nSaved:", OUTPUT_FILE)