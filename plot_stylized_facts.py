# Mean Real Markup Over Time by EDC
# Run python3 plot_stylized_facts.py to see the live mean_real_markup_by_edc.png

import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "markup_moments_real.csv"

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    dict(year=df["Year"], month=df["Month"], day=1)
)

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc]

    plt.plot(
        temp["Date"],
        temp["MeanMarkup"],
        label=edc
    )

plt.xlabel("Date")
plt.ylabel("Mean Real Markup (cents/kWh)")
plt.title("Mean Real Markup Over Time by EDC")
plt.legend()
plt.tight_layout()

plt.savefig(
    "mean_real_markup_by_edc.png",
    dpi=300
)

plt.show()



# distribution of real markup
# Run python3 plot_stylized_facts.py to see the live real_markup_distribution.png

markup = pd.read_csv("markup_full_real.csv")

plt.figure(figsize=(8, 5))

plt.hist(
    markup["real_markup"],
    bins=50,
    edgecolor="black"
)

plt.xlabel("Real Markup (cents/kWh)")
plt.ylabel("Number of Offers")
plt.title("Distribution of Real Markup")

plt.tight_layout()
plt.savefig(
    "real_markup_distribution.png",
    dpi=300
)

plt.show()


# distribution by EDC
# Run python3 plot_stylized_facts.py to see the live edc_distributions_npg (6 edcs)

edcs = sorted(markup["EDC"].unique())

output_dir = "/Users/cathy/markup-pipeline/edc_distributions_npg"

for edc in edcs:
    temp = markup[markup["EDC"] == edc]

    plt.figure(figsize=(8, 5))

    plt.hist(
        temp["real_markup"],
        bins=40,
        edgecolor="black"
    )

    plt.xlabel("Real Markup (cents/kWh)")
    plt.ylabel("Number of Offers")
    plt.title(f"Distribution of Real Markup - {edc}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"real_markup_distribution_{edc}.png"
        ),
        dpi=300
    )

    plt.close()

print("EDC distribution plots saved.")


# share above PTC over time by EDC
# Run python3 plot_stylized_facts.py to see the live share_above_ptc_by_edc.png

plt.figure(figsize=(9, 6))

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc]

    plt.plot(
        temp["Date"],
        temp["ShareAbovePTC"],
        label=edc
    )

plt.xlabel("Date")
plt.ylabel("Share Above PTC")
plt.title("Share of Offers Above PTC Over Time by EDC")
plt.legend()
plt.tight_layout()

plt.savefig(
    "share_above_ptc_by_edc.png",
    dpi=300
)

plt.show()



# price dispersion across markets

plt.figure(figsize=(8, 5))

plt.hist(
    df["Variance"],
    bins=40,
    edgecolor="black"
)

plt.xlabel("Variance of Real Markup")
plt.ylabel("Number of Markets")
plt.title("Distribution of Price Dispersion Across Markets")

plt.tight_layout()

plt.savefig(
    "variance_distribution.png",
    dpi=300
)

plt.show()



# price dispersion vs number of offers

plt.figure(figsize=(8, 5))

plt.scatter(
    df["n_offers"],
    df["Variance"],
    alpha=0.6
)

plt.xlabel("Number of Offers")
plt.ylabel("Variance of Real Markup")
plt.title("Price Dispersion vs Number of Offers")

plt.tight_layout()

plt.savefig(
    "variance_vs_offers.png",
    dpi=300
)

plt.show()



# mean markup vs number of offers

plt.figure(figsize=(8, 5))

plt.scatter(
    df["n_offers"],
    df["MeanMarkup"],
    alpha=0.6
)

plt.xlabel("Number of Offers")
plt.ylabel("Mean Real Markup (cents/kWh)")
plt.title("Mean Real Markup vs Number of Offers")

plt.tight_layout()

plt.savefig(
    "mean_markup_vs_offers.png",
    dpi=300
)

plt.show()



# summary of price dispersion by EDC

dispersion_summary = (
    df.groupby("EDC")["Variance"]
    .agg(
        Mean="mean",
        Median="median",
        Q10=lambda x: x.quantile(0.10),
        Q90=lambda x: x.quantile(0.90)
    )
    .reset_index()
)

dispersion_summary.to_csv(
    "dispersion_summary_by_edc.csv",
    index=False
)

print("\nPrice dispersion summary by EDC:")
print(dispersion_summary)



# markup percentiles over time

plt.figure(figsize=(9, 6))

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc]

    plt.plot(
        temp["Date"],
        temp["Q50"],
        label=edc
    )

plt.xlabel("Date")
plt.ylabel("Median Real Markup (cents/kWh)")
plt.title("Median Real Markup Over Time by EDC")
plt.legend()

plt.tight_layout()

plt.savefig(
    "median_real_markup_by_edc.png",
    dpi=300
)

plt.show()



# Q10 over time

plt.figure(figsize=(9, 6))

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc]

    plt.plot(
        temp["Date"],
        temp["Q10"],
        label=edc
    )

plt.xlabel("Date")
plt.ylabel("Q10 Real Markup (cents/kWh)")
plt.title("10th Percentile of Real Markup Over Time by EDC")
plt.legend()

plt.tight_layout()

plt.savefig(
    "q10_real_markup_by_edc.png",
    dpi=300
)

plt.show()



# Q90 over time

plt.figure(figsize=(9, 6))

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc]

    plt.plot(
        temp["Date"],
        temp["Q90"],
        label=edc
    )

plt.xlabel("Date")
plt.ylabel("Q90 Real Markup (cents/kWh)")
plt.title("90th Percentile of Real Markup Over Time by EDC")
plt.legend()

plt.tight_layout()

plt.savefig(
    "q90_real_markup_by_edc.png",
    dpi=300
)

plt.show()



# persistence of mean markup

import statsmodels.api as sm

persistence_rows = []

for edc in sorted(df["EDC"].unique()):
    temp = df[df["EDC"] == edc].sort_values("Date").copy()

    temp["LagMeanMarkup"] = temp["MeanMarkup"].shift(1)
    temp = temp.dropna(subset=["MeanMarkup", "LagMeanMarkup"])

    X = sm.add_constant(temp["LagMeanMarkup"])
    y = temp["MeanMarkup"]

    model = sm.OLS(y, X).fit()

    persistence_rows.append({
        "EDC": edc,
        "rho": model.params["LagMeanMarkup"],
        "intercept": model.params["const"],
        "R_squared": model.rsquared,
        "p_value": model.pvalues["LagMeanMarkup"],
        "n_obs": int(model.nobs)
    })

persistence = pd.DataFrame(persistence_rows)

persistence = persistence.round({
    "rho": 4,
    "intercept": 4,
    "R_squared": 4,
    "p_value": 4
})

persistence.to_csv(
    "markup_persistence_by_edc.csv",
    index=False
)

print("\nMarkup persistence by EDC:")
print(persistence)

print("\nAll stylized facts completed.")