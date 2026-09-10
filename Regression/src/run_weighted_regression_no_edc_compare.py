import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "regression_data.csv"
output_file = DATA_DIR / "weighted_regression_no_edc_compare.txt"

# Load data
df = pd.read_csv(input_file)

# Keep only rows needed for regression
df = df.dropna(
    subset=[
        "real_markup",
        "real_RT_Average",
        "weight",
        "Year"
    ]
)

# Weighted Least Squares
# Weight = 1 / number of offers in each EDC-month
# Year fixed effects are included
# EDC fixed effects are not included, so there is no reference EDC
model = smf.wls(
    formula="real_markup ~ real_RT_Average + C(Year)",
    data=df,
    weights=df["weight"]
).fit(cov_type="HC1")

print(model.summary())

# Save regression results
with open(output_file, "w") as f:
    f.write(model.summary().as_text())

print()
print("Regression completed.")
print("Number of observations:", int(model.nobs))
print("R-squared:", round(model.rsquared, 4))
print("RT coefficient:", round(model.params["real_RT_Average"], 4))
print("RT p-value:", round(model.pvalues["real_RT_Average"], 4))
print("Results saved to:", output_file)