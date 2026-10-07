import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PRESENTATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PRESENTATION_DIR.parent
DATA_DIR = PROJECT_DIR / "Regression" / "data"
OUTPUT_DIR = PRESENTATION_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OFFERS_FILE = DATA_DIR / "markup_full_real.csv"
RT_FILE = DATA_DIR / "rt_pa_all_years_summary.csv"


# ============================================================
# EDC configuration
#
# File names use three-letter acronyms.
# Titles use full EDC names.
# ============================================================

EDC_CONFIG = {
    "PPL": {
        "tla": "PPL",
        "full_name": "PPL Electric Utilities"
    },
    "PECO": {
        "tla": "PECO",
        "full_name": "PECO Energy Company"
    },
    "DUQ": {
        "tla": "DUQ",
        "full_name": "Duquesne Light"
    }
}

EDCS = list(EDC_CONFIG.keys())


# ============================================================
# Helper function
# ============================================================

def add_date(df):

    df = df.copy()

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="raise"
    ).astype(int)

    df["Month"] = pd.to_numeric(
        df["Month"],
        errors="raise"
    ).astype(int)

    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=1
        )
    )

    return df


# ============================================================
# 1. Load EGS / PTC data
# ============================================================

print("Loading EGS and PTC data...")

offers = pd.read_csv(
    OFFERS_FILE
)

offers = add_date(
    offers
)

offers = offers[
    offers["EDC"].isin(EDCS)
].copy()

offers["real_price"] = pd.to_numeric(
    offers["real_price"],
    errors="coerce"
)

offers["real_PTC"] = pd.to_numeric(
    offers["real_PTC"],
    errors="coerce"
)

print(
    "EGS/PTC coverage:",
    offers["Date"].min(),
    "to",
    offers["Date"].max()
)

print(
    "Offer rows:",
    len(offers)
)


# ============================================================
# 2. Load complete PJM RT monthly data
#
# rt_pa_all_years_summary.csv already contains:
#
# Year
# Month
# EDC
# Average
# Median
#
# The PJM pipeline calculated these monthly values from
# hourly RT LMP observations.
#
# IMPORTANT:
# This file is assumed to already be in cents/kWh.
# Do NOT divide it by 10 again here.
# ============================================================

print()
print("Loading complete PJM RT data...")

rt = pd.read_csv(
    RT_FILE
)

rt = add_date(
    rt
)

rt = rt[
    rt["EDC"].isin(EDCS)
].copy()

rt["Average"] = pd.to_numeric(
    rt["Average"],
    errors="coerce"
)

rt["Median"] = pd.to_numeric(
    rt["Median"],
    errors="coerce"
)


# Check duplicate EDC-month records

duplicate_rt = rt.duplicated(
    subset=[
        "Year",
        "Month",
        "EDC"
    ]
)

if duplicate_rt.any():

    raise ValueError(
        "Duplicate EDC-month records found "
        "in rt_pa_all_years_summary.csv"
    )


rt_monthly = (
    rt[
        [
            "Year",
            "Month",
            "EDC",
            "Date",
            "Average",
            "Median"
        ]
    ]
    .rename(
        columns={
            "Average": "PJM_RT_Average",
            "Median": "PJM_RT_Median"
        }
    )
    .sort_values(
        [
            "EDC",
            "Date"
        ]
    )
    .reset_index(drop=True)
)


print(
    "PJM coverage:",
    rt_monthly["Date"].min(),
    "to",
    rt_monthly["Date"].max()
)

print(
    "PJM rows:",
    len(rt_monthly)
)


# ============================================================
# 3. Monthly EGS Average / Median
#
# Individual EGS offers are summarized to monthly
# average and monthly median.
#
# The figures show one monthly marker connected by lines,
# matching the presentation style used in Scott's figure.
# ============================================================

egs_monthly = (
    offers
    .dropna(
        subset=["real_price"]
    )
    .groupby(
        [
            "Year",
            "Month",
            "EDC"
        ],
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
        EGS_Count=(
            "real_price",
            "size"
        )
    )
)

egs_monthly = add_date(
    egs_monthly
)


# ============================================================
# 4. Monthly PTC
#
# PTC is repeated across EGS offer rows.
#
# Remove exact duplicate PTC observations first so the
# number of EGS offers does not weight the PTC.
# ============================================================

ptc_unique = (
    offers[
        [
            "Year",
            "Month",
            "EDC",
            "real_PTC"
        ]
    ]
    .dropna(
        subset=["real_PTC"]
    )
    .drop_duplicates()
)


ptc_monthly = (
    ptc_unique
    .groupby(
        [
            "Year",
            "Month",
            "EDC"
        ],
        as_index=False
    )
    .agg(
        PTC_Average=(
            "real_PTC",
            "mean"
        ),
        PTC_Median=(
            "real_PTC",
            "median"
        ),
        PTC_Unique_Values=(
            "real_PTC",
            "nunique"
        )
    )
)

ptc_monthly = add_date(
    ptc_monthly
)


# Check months containing multiple PTC values

ptc_conflicts = ptc_monthly[
    ptc_monthly["PTC_Unique_Values"] > 1
]

if not ptc_conflicts.empty:

    print()
    print(
        "WARNING: Multiple distinct PTC values "
        "found for some EDC-months."
    )

    print(
        "These months use the unweighted "
        "mean/median of distinct PTC values."
    )

    print()

    print(
        ptc_conflicts[
            [
                "Year",
                "Month",
                "EDC",
                "PTC_Average",
                "PTC_Median",
                "PTC_Unique_Values"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 5. Build complete monthly dataset
#
# OUTER merges are intentional.
#
# PJM can therefore remain available for 2015-2026 even
# when EGS/PTC observations do not exist.
# ============================================================

monthly = (
    egs_monthly[
        [
            "Year",
            "Month",
            "EDC",
            "EGS_Average",
            "EGS_Median",
            "EGS_Count"
        ]
    ]
    .merge(
        ptc_monthly[
            [
                "Year",
                "Month",
                "EDC",
                "PTC_Average",
                "PTC_Median",
                "PTC_Unique_Values"
            ]
        ],
        on=[
            "Year",
            "Month",
            "EDC"
        ],
        how="outer"
    )
    .merge(
        rt_monthly[
            [
                "Year",
                "Month",
                "EDC",
                "PJM_RT_Average",
                "PJM_RT_Median"
            ]
        ],
        on=[
            "Year",
            "Month",
            "EDC"
        ],
        how="outer"
    )
)

monthly = add_date(
    monthly
)

monthly = (
    monthly
    .sort_values(
        [
            "EDC",
            "Date"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 6. Save supporting monthly data
# ============================================================

monthly.to_csv(
    OUTPUT_DIR / "B_a_monthly_data.csv",
    index=False
)

print()
print(
    "Created:",
    OUTPUT_DIR / "B_a_monthly_data.csv"
)


# ============================================================
# 7. Coverage report
# ============================================================

print()
print(
    "========================================"
)

print(
    "MONTHLY DATA COVERAGE"
)

print(
    "========================================"
)


for edc in EDCS:

    temp = monthly[
        monthly["EDC"] == edc
    ]

    print()
    print(edc)

    print(
        "Full range:",
        temp["Date"].min(),
        "to",
        temp["Date"].max()
    )

    print(
        "PJM months:",
        temp["PJM_RT_Average"]
        .notna()
        .sum()
    )

    print(
        "EGS months:",
        temp["EGS_Average"]
        .notna()
        .sum()
    )

    print(
        "PTC months:",
        temp["PTC_Average"]
        .notna()
        .sum()
    )


# ============================================================
# 8. Average figure
#
# Scott-style axes:
#
# LEFT:
# EGS + PTC
# $/kWh
#
# RIGHT:
# PJM wholesale
# $/MWh
#
# Original stored data remains unchanged.
# Unit conversion happens only for plotting.
# ============================================================

for edc in EDCS:

    config = EDC_CONFIG[edc]

    tla = config["tla"]
    full_name = config["full_name"]

    temp = (
        monthly[
            monthly["EDC"] == edc
        ]
        .sort_values("Date")
        .copy()
    )


    # --------------------------------------------------------
    # Convert units only for plotting
    #
    # cents/kWh -> $/kWh
    #
    # 1 cent/kWh = 0.01 $/kWh
    #
    # cents/kWh -> $/MWh
    #
    # 1 cent/kWh = 10 $/MWh
    # --------------------------------------------------------

    temp["EGS_Average_plot"] = (
        temp["EGS_Average"]
        / 100
    )

    temp["PTC_Average_plot"] = (
        temp["PTC_Average"]
        / 100
    )

    temp["PJM_RT_Average_plot"] = (
        temp["PJM_RT_Average"]
        * 10
    )


    # --------------------------------------------------------
    # Create dual-axis figure
    # --------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )


    # --------------------------------------------------------
    # LEFT Y-axis
    # PTC
    # --------------------------------------------------------

    line_ptc, = ax1.plot(
        temp["Date"],
        temp["PTC_Average_plot"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="PTC"
    )


    # --------------------------------------------------------
    # LEFT Y-axis
    # EGS Average
    # --------------------------------------------------------

    line_egs, = ax1.plot(
        temp["Date"],
        temp["EGS_Average_plot"],
        marker="x",
        markersize=4,
        linewidth=1.2,
        label="EGS avg"
    )


    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Retail price ($/kWh)"
    )


    # --------------------------------------------------------
    # RIGHT Y-axis
    # PJM wholesale
    # --------------------------------------------------------

    ax2 = ax1.twinx()

    line_pjm, = ax2.plot(
        temp["Date"],
        temp["PJM_RT_Average_plot"],
        linewidth=1.2,
        alpha=0.65,
        color="green",
        label="PJM RT"
    )


    ax2.set_ylabel(
        "PJM wholesale price ($/MWh)",
        color="green"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="green"
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax1.set_title(
        f"{full_name}: "
        "Average EGS Rate vs PTC vs PJM Wholesale"
    )


    # --------------------------------------------------------
    # Legend
    # Match Scott-style retail legend
    # --------------------------------------------------------

    ax1.legend(
        handles=[
            line_ptc,
            line_egs
        ],
        loc="upper left"
    )


    ax1.grid(
        alpha=0.2
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{tla}_average_egs_ptc_rt.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Created:",
        filename
    )


# ============================================================
# 9. Median figure
#
# Same dual-axis structure as Average.
# ============================================================

for edc in EDCS:

    config = EDC_CONFIG[edc]

    tla = config["tla"]
    full_name = config["full_name"]

    temp = (
        monthly[
            monthly["EDC"] == edc
        ]
        .sort_values("Date")
        .copy()
    )


    # --------------------------------------------------------
    # Unit conversion for plotting
    # --------------------------------------------------------

    temp["EGS_Median_plot"] = (
        temp["EGS_Median"]
        / 100
    )

    temp["PTC_Median_plot"] = (
        temp["PTC_Median"]
        / 100
    )

    temp["PJM_RT_Median_plot"] = (
        temp["PJM_RT_Median"]
        * 10
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )


    # --------------------------------------------------------
    # LEFT Y-axis
    # PTC
    # --------------------------------------------------------

    line_ptc, = ax1.plot(
        temp["Date"],
        temp["PTC_Median_plot"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="PTC"
    )


    # --------------------------------------------------------
    # LEFT Y-axis
    # EGS Median
    # --------------------------------------------------------

    line_egs, = ax1.plot(
        temp["Date"],
        temp["EGS_Median_plot"],
        marker="x",
        markersize=4,
        linewidth=1.2,
        label="EGS median"
    )


    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Retail price ($/kWh)"
    )


    # --------------------------------------------------------
    # RIGHT Y-axis
    # PJM RT Median
    # --------------------------------------------------------

    ax2 = ax1.twinx()

    line_pjm, = ax2.plot(
        temp["Date"],
        temp["PJM_RT_Median_plot"],
        linewidth=1.2,
        alpha=0.65,
        color="green",
        label="PJM RT"
    )


    ax2.set_ylabel(
        "PJM wholesale price ($/MWh)",
        color="green"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="green"
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax1.set_title(
        f"{full_name}: "
        "Median EGS Rate vs PTC vs PJM Wholesale"
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax1.legend(
        handles=[
            line_ptc,
            line_egs
        ],
        loc="upper left"
    )


    ax1.grid(
        alpha=0.2
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{tla}_median_egs_ptc_rt.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Created:",
        filename
    )


# ============================================================
# 10. Save EDC-level monthly supporting data
# ============================================================

for edc in EDCS:

    tla = EDC_CONFIG[edc]["tla"]

    temp = (
        monthly[
            monthly["EDC"] == edc
        ]
        .sort_values("Date")
        .copy()
    )

    temp.to_csv(
        OUTPUT_DIR
        / f"{tla}_egs_ptc_rt.csv",
        index=False
    )


# ============================================================
# 11. Above / Below PTC classification
#
# Classification is performed at the individual offer level.
# ============================================================

classified = (
    offers
    .dropna(
        subset=[
            "real_price",
            "real_PTC"
        ]
    )
    .copy()
)


classified["Above_PTC"] = (
    classified["real_price"]
    > classified["real_PTC"]
)


classified["Below_PTC"] = (
    classified["real_price"]
    < classified["real_PTC"]
)


classified["Equal_PTC"] = (
    classified["real_price"]
    == classified["real_PTC"]
)


# ============================================================
# 12. Monthly Above / Below counts and shares
# ============================================================

monthly_bd = (
    classified
    .groupby(
        [
            "Year",
            "Month",
            "EDC"
        ],
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


# ============================================================
# 13. Merge complete PJM into Above / Below data
#
# OUTER merge preserves the full PJM history.
# ============================================================

monthly_bd = monthly_bd.merge(
    rt_monthly[
        [
            "Year",
            "Month",
            "EDC",
            "PJM_RT_Average"
        ]
    ],
    on=[
        "Year",
        "Month",
        "EDC"
    ],
    how="outer"
)


monthly_bd = add_date(
    monthly_bd
)


monthly_bd = (
    monthly_bd
    .sort_values(
        [
            "EDC",
            "Date"
        ]
    )
    .reset_index(drop=True)
)


monthly_bd.to_csv(
    OUTPUT_DIR
    / "B_d_above_below_ptc_data.csv",
    index=False
)


# ============================================================
# 14. Above / Below PTC share figures
#
# LEFT:
# Share of EGS offers
#
# RIGHT:
# PJM wholesale $/MWh
# ============================================================

for edc in EDCS:

    config = EDC_CONFIG[edc]

    tla = config["tla"]
    full_name = config["full_name"]

    temp = (
        monthly_bd[
            monthly_bd["EDC"] == edc
        ]
        .sort_values("Date")
        .copy()
    )


    # Convert PJM:
    # cents/kWh -> $/MWh

    temp["PJM_RT_Average_plot"] = (
        temp["PJM_RT_Average"]
        * 10
    )


    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )


    # --------------------------------------------------------
    # LEFT Y-axis
    # Above / Below shares
    # --------------------------------------------------------

    line_above, = ax1.plot(
        temp["Date"],
        temp["Share_Above_PTC"] * 100,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="EGS Above PTC (%)"
    )


    line_below, = ax1.plot(
        temp["Date"],
        temp["Share_Below_PTC"] * 100,
        marker="x",
        markersize=4,
        linewidth=1.2,
        label="EGS Below PTC (%)"
    )


    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Share of EGS Offers (%)"
    )

    ax1.set_ylim(
        0,
        100
    )


    # --------------------------------------------------------
    # RIGHT Y-axis
    # PJM wholesale
    # --------------------------------------------------------

    ax2 = ax1.twinx()

    line_pjm, = ax2.plot(
        temp["Date"],
        temp["PJM_RT_Average_plot"],
        linewidth=1.2,
        alpha=0.65,
        color="green",
        label="PJM RT"
    )


    ax2.set_ylabel(
        "PJM wholesale price ($/MWh)",
        color="green"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="green"
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax1.set_title(
        f"{full_name}: "
        "EGS Offers Above/Below PTC "
        "vs PJM Wholesale"
    )


    # --------------------------------------------------------
    # Combined legend
    # --------------------------------------------------------

    ax1.legend(
        handles=[
            line_above,
            line_below,
            line_pjm
        ],
        loc="upper left"
    )


    ax1.grid(
        alpha=0.2
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{tla}_above_below_ptc_rt.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Created:",
        filename
    )


# ============================================================
# 15. Monthly counts Above / Below PTC
# ============================================================

for edc in EDCS:

    config = EDC_CONFIG[edc]

    tla = config["tla"]
    full_name = config["full_name"]

    temp = (
        monthly_bd[
            monthly_bd["EDC"] == edc
        ]
        .sort_values("Date")
        .copy()
    )


    fig, ax = plt.subplots(
        figsize=(12, 6)
    )


    ax.plot(
        temp["Date"],
        temp["Above_PTC"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Offers Above PTC"
    )


    ax.plot(
        temp["Date"],
        temp["Below_PTC"],
        marker="x",
        markersize=4,
        linewidth=1.2,
        label="Offers Below PTC"
    )


    ax.set_title(
        f"{full_name}: "
        "Monthly EGS Offer Counts "
        "Above and Below PTC"
    )


    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Number of Offers"
    )


    ax.legend(
        loc="upper left"
    )


    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = (
        f"{tla}_ptc_like_above_below_ptc_rt.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Created:",
        filename
    )


# ============================================================
# 16. Final summary
# ============================================================

print()
print(
    "========================================"
)

print(
    "PRESENTATION FIGURES COMPLETE"
)

print(
    "========================================"
)


for edc in EDCS:

    tla = EDC_CONFIG[edc]["tla"]

    print()
    print(
        f"{edc} ({tla})"
    )

    print(
        f"  {tla}_average_egs_ptc_rt.png"
    )

    print(
        f"  {tla}_median_egs_ptc_rt.png"
    )

    print(
        f"  {tla}_above_below_ptc_rt.png"
    )

    print(
        f"  {tla}_ptc_like_above_below_ptc_rt.png"
    )

    print(
        f"  {tla}_egs_ptc_rt.csv"
    )


print()
print(
    "Supporting CSV files:"
)

print(
    "  B_a_monthly_data.csv"
)

print(
    "  B_d_above_below_ptc_data.csv"
)


print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)