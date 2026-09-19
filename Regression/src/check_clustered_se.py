import pandas as pd
import statsmodels.formula.api as smf

# Load final regression data
df = pd.read_csv("../data/regression_data.csv")

# Create EDC-month cluster ID
df["cluster"] = (
    df["EDC"].astype(str)
    + "_"
    + df["Year"].astype(str)
    + "_"
    + df["Month"].astype(str)
)

# Same regression specification
formula = "real_markup ~ real_RT_Average + C(Year) + C(EDC)"

# Original HC1 standard errors
model_hc1 = smf.wls(
    formula=formula,
    data=df,
    weights=df["weight"]
).fit(cov_type="HC1")

# EDC-month clustered standard errors
model_cluster = smf.wls(
    formula=formula,
    data=df,
    weights=df["weight"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cluster"]}
)

# Compare the RT coefficient
variable = "real_RT_Average"

print("\nHC1 RESULTS")
print("------------------------------")
print("Coefficient:", model_hc1.params[variable])
print("Std. Error :", model_hc1.bse[variable])
print("p-value    :", model_hc1.pvalues[variable])
print("95% CI     :", model_hc1.conf_int().loc[variable].tolist())

print("\nEDC-MONTH CLUSTERED RESULTS")
print("------------------------------")
print("Coefficient:", model_cluster.params[variable])
print("Std. Error :", model_cluster.bse[variable])
print("p-value    :", model_cluster.pvalues[variable])
print("95% CI     :", model_cluster.conf_int().loc[variable].tolist())

print("\nNumber of observations:", len(df))
print("Number of EDC-month clusters:", df["cluster"].nunique())

# Save full clustered regression results
with open("../data/clustered_regression_results.txt", "w") as f:
    f.write(model_cluster.summary().as_text())