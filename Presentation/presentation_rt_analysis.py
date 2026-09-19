import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PRESENTATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PRESENTATION_DIR.parent
DATA_DIR = PROJECT_DIR / "Regression" / "data"
OUTPUT_DIR = PRESENTATION_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "regression_data.csv"


# --------------------------------------------------
# 1. Load final regression data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

needed = [
    "Year",
    "Month",
    "EDC",
    "real_markup",
    "real_RT_Average",
    "weight"
]

df = df.dropna(subset=needed).copy()

df["Year"] = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)

df["cluster"] = (
    df["EDC"].astype(str)
    + "_"
    + df["Year"].astype(str)
    + "_"
    + df["Month"].astype(str)
)

print("========================================")
print("FINAL RT REGRESSION SAMPLE")
print("========================================")
print("Observations:", len(df))
print("Years:", df["Year"].min(), "-", df["Year"].max())
print("EDCs:", sorted(df["EDC"].unique()))
print("EDC-month clusters:", df["cluster"].nunique())


# --------------------------------------------------
# 2. Check equal weighting by EDC-month
# --------------------------------------------------

weight_check = (
    df.groupby(["Year", "Month", "EDC"])["weight"]
      .sum()
      .reset_index(name="total_weight")
)

weight_check.to_csv(
    OUTPUT_DIR / "weight_check.csv",
    index=False
)

print("\n========================================")
print("WEIGHT CHECK")
print("========================================")
print("Minimum EDC-month total weight:",
      weight_check["total_weight"].min())
print("Maximum EDC-month total weight:",
      weight_check["total_weight"].max())


# --------------------------------------------------
# 3. Baseline RT regression
#
# Real Markup =
# RT wholesale price
# + Year Fixed Effects
# + EDC Fixed Effects
# --------------------------------------------------

baseline = smf.wls(
    formula="real_markup ~ real_RT_Average + C(Year) + C(EDC)",
    data=df,
    weights=df["weight"]
).fit(cov_type="HC1")

with open(
    OUTPUT_DIR / "baseline_rt_regression.txt",
    "w"
) as f:
    f.write(baseline.summary().as_text())

print("\n========================================")
print("BASELINE RT REGRESSION")
print("========================================")
print("RT coefficient:",
      round(baseline.params["real_RT_Average"], 4))
print("HC1 standard error:",
      round(baseline.bse["real_RT_Average"], 4))
print("p-value:",
      round(baseline.pvalues["real_RT_Average"], 4))
print("R-squared:",
      round(baseline.rsquared, 4))


# --------------------------------------------------
# 4. Same regression with EDC-month clustered SE
# --------------------------------------------------

clustered = smf.wls(
    formula="real_markup ~ real_RT_Average + C(Year) + C(EDC)",
    data=df,
    weights=df["weight"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cluster"]}
)

with open(
    OUTPUT_DIR / "clustered_rt_regression.txt",
    "w"
) as f:
    f.write(clustered.summary().as_text())

print("\n========================================")
print("CLUSTERED RT REGRESSION")
print("========================================")
print("RT coefficient:",
      round(clustered.params["real_RT_Average"], 4))
print("Clustered standard error:",
      round(clustered.bse["real_RT_Average"], 4))
print("p-value:",
      round(clustered.pvalues["real_RT_Average"], 4))
print("95% CI:",
      clustered.conf_int().loc[
          "real_RT_Average"
      ].round(4).tolist())


# --------------------------------------------------
# 5. HC1 vs clustered SE comparison
# --------------------------------------------------

comparison = pd.DataFrame({
    "Specification": [
        "HC1",
        "EDC-month clustered"
    ],
    "RT_Coefficient": [
        baseline.params["real_RT_Average"],
        clustered.params["real_RT_Average"]
    ],
    "Std_Error": [
        baseline.bse["real_RT_Average"],
        clustered.bse["real_RT_Average"]
    ],
    "P_Value": [
        baseline.pvalues["real_RT_Average"],
        clustered.pvalues["real_RT_Average"]
    ]
})

comparison = comparison.round(4)

comparison.to_csv(
    OUTPUT_DIR / "rt_se_comparison.csv",
    index=False
)


# --------------------------------------------------
# 6. Markup summary by EDC
# --------------------------------------------------

edc_summary = (
    df.groupby("EDC")
      .agg(
          Mean_Real_Markup=("real_markup", "mean"),
          Median_Real_Markup=("real_markup", "median"),
          Std_Real_Markup=("real_markup", "std"),
          Mean_Real_RT=("real_RT_Average", "mean"),
          N=("real_markup", "size")
      )
      .reset_index()
)

below_ptc = (
    df.assign(Below_PTC=df["real_markup"] < 0)
      .groupby("EDC")["Below_PTC"]
      .mean()
      .reset_index(name="Share_Below_PTC")
)

edc_summary = edc_summary.merge(
    below_ptc,
    on="EDC",
    how="left"
)

edc_summary["Share_Below_PTC"] *= 100

edc_summary = edc_summary.round(4)

edc_summary.to_csv(
    OUTPUT_DIR / "edc_summary.csv",
    index=False
)

print("\n========================================")
print("EDC SUMMARY")
print("========================================")
print(edc_summary.to_string(index=False))


# --------------------------------------------------
# 7. Separate RT slope for each EDC
# --------------------------------------------------

edc_results = []

for edc in sorted(df["EDC"].unique()):

    sub = df[df["EDC"] == edc].copy()

    model = smf.wls(
        formula="real_markup ~ real_RT_Average + C(Year)",
        data=sub,
        weights=sub["weight"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["cluster"]}
    )

    ci = model.conf_int().loc["real_RT_Average"]

    edc_results.append({
        "EDC": edc,
        "RT_Coefficient":
            model.params["real_RT_Average"],
        "Clustered_SE":
            model.bse["real_RT_Average"],
        "P_Value":
            model.pvalues["real_RT_Average"],
        "CI_Lower":
            ci.iloc[0],
        "CI_Upper":
            ci.iloc[1],
        "R_Squared":
            model.rsquared,
        "N":
            int(model.nobs)
    })

edc_results = pd.DataFrame(edc_results).round(4)

edc_results.to_csv(
    OUTPUT_DIR / "rt_regression_by_edc.csv",
    index=False
)

print("\n========================================")
print("RT REGRESSION BY EDC")
print("========================================")
print(edc_results.to_string(index=False))


# --------------------------------------------------
# 8. PPL / PECO / DUQ two-way comparison
#
# Allows the RT relationship to differ by EDC.
# PECO is the reference group.
# --------------------------------------------------

three = df[
    df["EDC"].isin(["PECO", "PPL", "DUQ"])
].copy()

three["EDC"] = pd.Categorical(
    three["EDC"],
    categories=["PECO", "PPL", "DUQ"]
)

three_way = smf.wls(
    formula=(
        "real_markup ~ "
        "real_RT_Average * C(EDC) "
        "+ C(Year)"
    ),
    data=three,
    weights=three["weight"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": three["cluster"]}
)

with open(
    OUTPUT_DIR / "ppl_peco_duq_interaction.txt",
    "w"
) as f:
    f.write(three_way.summary().as_text())

print("\n========================================")
print("PPL / PECO / DUQ TWO-WAY COMPARISON")
print("========================================")
print(three_way.summary())


# --------------------------------------------------
# 9. Recover RT slopes from interaction regression
# --------------------------------------------------

peco_slope = three_way.params["real_RT_Average"]

ppl_interaction = (
    "real_RT_Average:C(EDC)[T.PPL]"
)

duq_interaction = (
    "real_RT_Average:C(EDC)[T.DUQ]"
)

ppl_slope = (
    peco_slope
    + three_way.params.get(
        ppl_interaction, 0
    )
)

duq_slope = (
    peco_slope
    + three_way.params.get(
        duq_interaction, 0
    )
)

three_slopes = pd.DataFrame({
    "EDC": ["PECO", "PPL", "DUQ"],
    "Estimated_RT_Slope": [
        peco_slope,
        ppl_slope,
        duq_slope
    ]
}).round(4)

three_slopes.to_csv(
    OUTPUT_DIR / "ppl_peco_duq_rt_slopes.csv",
    index=False
)

print("\nEstimated RT slopes:")
print(three_slopes.to_string(index=False))


# --------------------------------------------------
# 10. Monthly EDC-level panel for presentation
#
# Collapse multiple retail offers so each
# EDC-month appears once.
# --------------------------------------------------

monthly = (
    df.groupby(
        ["Year", "Month", "EDC"],
        as_index=False
    )
    .apply(
        lambda x: pd.Series({
            "Mean_Real_Markup":
                np.average(
                    x["real_markup"],
                    weights=x["weight"]
                ),
            "Real_RT_Average":
                x["real_RT_Average"].iloc[0],
            "Number_of_Offers":
                len(x)
        }),
        include_groups=False
    )
    .reset_index(drop=True)
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_edc_panel.csv",
    index=False
)


# --------------------------------------------------
# 11. Figure: mean real markup by EDC
# --------------------------------------------------

plot_summary = (
    monthly.groupby("EDC")["Mean_Real_Markup"]
           .mean()
           .sort_values()
)

plt.figure(figsize=(8, 5))
plot_summary.plot(kind="bar")

plt.axhline(0, linewidth=1)

plt.ylabel("Mean Real Markup")
plt.xlabel("EDC")
plt.title("Mean Real Retail Markup by EDC")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mean_real_markup_by_edc.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 12. Figure: RT coefficient by EDC
# --------------------------------------------------

plot_coef = edc_results.sort_values(
    "RT_Coefficient"
).copy()

x = np.arange(len(plot_coef))

lower_error = (
    plot_coef["RT_Coefficient"]
    - plot_coef["CI_Lower"]
)

upper_error = (
    plot_coef["CI_Upper"]
    - plot_coef["RT_Coefficient"]
)

plt.figure(figsize=(8, 5))

plt.errorbar(
    x,
    plot_coef["RT_Coefficient"],
    yerr=[
        lower_error,
        upper_error
    ],
    fmt="o",
    capsize=4
)

plt.axhline(0, linewidth=1)

plt.xticks(
    x,
    plot_coef["EDC"]
)

plt.ylabel("RT Coefficient")
plt.xlabel("EDC")
plt.title(
    "Relationship Between RT Wholesale Price "
    "and Real Retail Markup"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rt_coefficient_by_edc.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# PPL / PECO / DUQ markup over time
# ------------------------------------------------------------

plot_data = pd.read_csv(OUTPUT_DIR / "monthly_edc_panel.csv")

plot_data["Date"] = pd.to_datetime(
    plot_data["Year"].astype(str) + "-" +
    plot_data["Month"].astype(str) + "-01"
)

compare_edcs = ["PPL", "PECO", "DUQ"]

plt.figure(figsize=(15, 7.5))

for edc in compare_edcs:
    temp = plot_data[plot_data["EDC"] == edc].copy()
    temp = temp.sort_values("Date")

    plt.plot(
        temp["Date"],
        temp["Mean_Real_Markup"],
        label=edc
    )

plt.axhline(0, linewidth=1)

plt.title("Real Retail Markup: PPL, PECO, and DUQ")
plt.xlabel("Date")
plt.ylabel("Mean Real Markup")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ppl_peco_duq_markup_over_time.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 14. Presentation summary
# --------------------------------------------------

summary_file = OUTPUT_DIR / "presentation_summary.txt"

with open(summary_file, "w") as f:

    f.write(
        "RT WHOLESALE / RETAIL MARKUP ANALYSIS\n"
    )
    f.write(
        "=====================================\n\n"
    )

    f.write(
        f"Sample: {len(df):,} retail offer observations\n"
    )

    f.write(
        f"Period: {df['Year'].min()}-"
        f"{df['Year'].max()}\n"
    )

    f.write(
        "EDCs: "
        + ", ".join(
            sorted(df["EDC"].unique())
        )
        + "\n"
    )

    f.write(
        f"EDC-month clusters: "
        f"{df['cluster'].nunique()}\n\n"
    )

    f.write(
        "Baseline weighted regression:\n"
    )

    f.write(
        "real_markup ~ real_RT_Average "
        "+ Year FE + EDC FE\n\n"
    )

    f.write(
        f"RT coefficient: "
        f"{baseline.params['real_RT_Average']:.4f}\n"
    )

    f.write(
        f"HC1 SE: "
        f"{baseline.bse['real_RT_Average']:.4f}\n"
    )

    f.write(
        f"HC1 p-value: "
        f"{baseline.pvalues['real_RT_Average']:.4f}\n\n"
    )

    f.write(
        "EDC-month clustered standard errors:\n"
    )

    f.write(
        f"Clustered SE: "
        f"{clustered.bse['real_RT_Average']:.4f}\n"
    )

    f.write(
        f"Clustered p-value: "
        f"{clustered.pvalues['real_RT_Average']:.4f}\n\n"
    )

    f.write(
        f"R-squared: "
        f"{baseline.rsquared:.4f}\n"
    )


print("\n========================================")
print("DONE")
print("========================================")
print("All presentation files saved in:")
print(OUTPUT_DIR)