import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PRESENTATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PRESENTATION_DIR.parent
DATA_DIR = PROJECT_DIR / "Regression" / "data"
OUTPUT_DIR = PRESENTATION_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OFFERS_FILE = DATA_DIR / "markup_full_real.csv"
RT_FILE = DATA_DIR / "regression_data.csv"

EDCS = ["PPL", "PECO", "DUQ"]


# --------------------------------------------------
# 1. Load retail / PTC data
# --------------------------------------------------

offers = pd.read_csv(OFFERS_FILE)

offers["Year"] = offers["Year"].astype(int)
offers["Month"] = offers["Month"].astype(int)

offers["Date"] = pd.to_datetime(
    dict(
        year=offers["Year"],
        month=offers["Month"],
        day=1
    )
)

offers = offers[
    offers["EDC"].isin(EDCS)
].copy()

offers = offers.dropna(
    subset=[
        "real_price",
        "real_PTC"
    ]
)


# --------------------------------------------------
# 2. Load final processed RT data
# --------------------------------------------------

rt = pd.read_csv(RT_FILE)

rt["Year"] = rt["Year"].astype(int)
rt["Month"] = rt["Month"].astype(int)

rt = rt[
    rt["EDC"].isin(EDCS)
].copy()


# --------------------------------------------------
# 3. One RT value per EDC-month
#
# real_RT_Average and real_RT_Median are already
# calculated in the regression pipeline.
# Do not calculate inflation adjustment again here.
# --------------------------------------------------

rt_monthly = (
    rt.groupby(
        ["Year", "Month", "EDC"],
        as_index=False
    )
    .agg(
        PJM_RT_Average=(
            "real_RT_Average",
            "first"
        ),
        PJM_RT_Median=(
            "real_RT_Median",
            "first"
        )
    )
)


# --------------------------------------------------
# 4. B.a retail / PTC monthly data
# --------------------------------------------------

retail_monthly = (
    offers.groupby(
        ["Year", "Month", "Date", "EDC"],
        as_index=False
    )
    .agg(
        EGS_Average=(
            "real_price",
            "mean"
        ),
        EGS_Median=(
            "real_price",
            "median"
        ),
        PTC_Average=(
            "real_PTC",
            "mean"
        ),
        PTC_Median=(
            "real_PTC",
            "median"
        )
    )
)


# --------------------------------------------------
# 5. Merge retail/PTC with processed RT
#
# LEFT merge is important:
# retail/PTC months are preserved even if an RT
# value is unavailable for a particular month.
# --------------------------------------------------

monthly_ba = retail_monthly.merge(
    rt_monthly,
    on=["Year", "Month", "EDC"],
    how="left"
)

monthly_ba.to_csv(
    OUTPUT_DIR / "B_a_monthly_data.csv",
    index=False
)


print("========================================")
print("B.a DATA")
print("========================================")

for edc in EDCS:

    temp = monthly_ba[
        monthly_ba["EDC"] == edc
    ]

    print(
        edc,
        "| months:",
        len(temp),
        "| RT months:",
        temp["PJM_RT_Average"].notna().sum()
    )


# --------------------------------------------------
# 6. B.a figures
# --------------------------------------------------

for edc in EDCS:

    temp = monthly_ba[
        monthly_ba["EDC"] == edc
    ].sort_values("Date")


    # Average
    plt.figure(figsize=(12, 6))

    plt.plot(
        temp["Date"],
        temp["EGS_Average"],
        label="Average EGS Rate"
    )

    plt.plot(
        temp["Date"],
        temp["PTC_Average"],
        label="Average PTC"
    )

    plt.plot(
        temp["Date"],
        temp["PJM_RT_Average"],
        label="PJM RT Average"
    )

    plt.title(
        f"{edc}: Average EGS Rate, PTC, and PJM RT Price"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Inflation-Adjusted Price (cents/kWh)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{edc}_average_egs_ptc_rt.png",
        dpi=300
    )

    plt.close()


    # Median
    plt.figure(figsize=(12, 6))

    plt.plot(
        temp["Date"],
        temp["EGS_Median"],
        label="Median EGS Rate"
    )

    plt.plot(
        temp["Date"],
        temp["PTC_Median"],
        label="Median PTC"
    )

    plt.plot(
        temp["Date"],
        temp["PJM_RT_Median"],
        label="PJM RT Median"
    )

    plt.title(
        f"{edc}: Median EGS Rate, PTC, and PJM RT Price"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Inflation-Adjusted Price (cents/kWh)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{edc}_median_egs_ptc_rt.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 7. B.d
#
# All EGS offers above / below PTC.
# This calculation uses the complete retail data,
# not the regression sample.
# --------------------------------------------------

offers["Above_PTC"] = (
    offers["real_price"]
    > offers["real_PTC"]
)

offers["Below_PTC"] = (
    offers["real_price"]
    < offers["real_PTC"]
)

offers["Equal_PTC"] = (
    offers["real_price"]
    == offers["real_PTC"]
)


monthly_bd = (
    offers.groupby(
        ["Year", "Month", "Date", "EDC"],
        as_index=False
    )
    .agg(
        Total_Offers=(
            "real_price",
            "size"
        ),
        Above_PTC=(
            "Above_PTC",
            "sum"
        ),
        Below_PTC=(
            "Below_PTC",
            "sum"
        ),
        Equal_PTC=(
            "Equal_PTC",
            "sum"
        )
    )
)


monthly_bd["Share_Above_PTC"] = (
    monthly_bd["Above_PTC"]
    / monthly_bd["Total_Offers"]
)

monthly_bd["Share_Below_PTC"] = (
    monthly_bd["Below_PTC"]
    / monthly_bd["Total_Offers"]
)

monthly_bd["Share_Equal_PTC"] = (
    monthly_bd["Equal_PTC"]
    / monthly_bd["Total_Offers"]
)


# Add processed RT after above/below calculation
monthly_bd = monthly_bd.merge(
    rt_monthly[
        [
            "Year",
            "Month",
            "EDC",
            "PJM_RT_Average"
        ]
    ],
    on=["Year", "Month", "EDC"],
    how="left"
)


monthly_bd.to_csv(
    OUTPUT_DIR
    / "B_d_above_below_ptc_data.csv",
    index=False
)


# --------------------------------------------------
# 8. B.d figures
# --------------------------------------------------

for edc in EDCS:

    temp = monthly_bd[
        monthly_bd["EDC"] == edc
    ].sort_values("Date")

    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )

    ax1.plot(
        temp["Date"],
        temp["Share_Above_PTC"] * 100,
        label="Offers Above PTC"
    )

    ax1.plot(
        temp["Date"],
        temp["Share_Below_PTC"] * 100,
        label="Offers Below PTC"
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Share of Offers (%)")
    ax1.set_ylim(0, 100)


    ax2 = ax1.twinx()

    ax2.plot(
        temp["Date"],
        temp["PJM_RT_Average"],
        linestyle="--",
        label="PJM RT Average"
    )

    ax2.set_ylabel(
        "Inflation-Adjusted PJM RT Price "
        "(cents/kWh)"
    )


    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )


    plt.title(
        f"{edc}: Offers Above/Below PTC "
        "and PJM RT Price"
    )

    fig.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{edc}_above_below_ptc_rt.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 9. Done
# --------------------------------------------------

print("\n========================================")
print("DONE")
print("========================================")

print("\nB.a:")
for edc in EDCS:
    print(
        f"  {edc}_average_egs_ptc_rt.png"
    )
    print(
        f"  {edc}_median_egs_ptc_rt.png"
    )

print("\nB.d:")
for edc in EDCS:
    print(
        f"  {edc}_above_below_ptc_rt.png"
    )

print("\nSupporting data:")
print("  B_a_monthly_data.csv")
print("  B_d_above_below_ptc_data.csv")

print("\nAll files saved in:")
print(OUTPUT_DIR)