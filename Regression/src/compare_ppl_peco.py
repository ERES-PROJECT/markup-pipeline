import pandas as pd
import statsmodels.formula.api as smf

# Load regression data
df = pd.read_csv("../data/regression_data.csv")

# Keep only PPL and PECO
df = df[df["EDC"].isin(["PPL", "PECO"])].copy()

# Use PECO as the reference group
df["EDC"] = pd.Categorical(
    df["EDC"],
    categories=["PECO", "PPL"]
)

# WLS comparison
# Allow both the RT slope and year effects to differ between PPL and PECO
model = smf.wls(
    formula=(
        "real_markup ~ "
        "C(EDC) * (real_RT_Average + C(Year))"
    ),
    data=df,
    weights=df["weight"]
).fit(cov_type="HC1")

print(model.summary())

# Save results
with open("../data/ppl_peco_interaction_results.txt", "w") as f:
    f.write(model.summary().as_text())