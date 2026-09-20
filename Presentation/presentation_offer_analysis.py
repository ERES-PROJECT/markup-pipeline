import os
import re
import json
import glob
import zipfile

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKUP_DIR = os.path.dirname(BASE_DIR)
PROJECTS_DIR = os.path.dirname(MARKUP_DIR)

WATTBUY_DIR = os.path.join(
    PROJECTS_DIR,
    "wattbuy-pipeline"
)

PA_SWITCH_DIR = os.path.join(
    PROJECTS_DIR,
    "pa-switch-pipeline"
)

MARKUP_DATA_DIR = os.path.join(
    MARKUP_DIR,
    "Regression",
    "data"
)

MARKUP_FILE = os.path.join(
    MARKUP_DATA_DIR,
    "markup_full_real.csv"
)

RT_FILE = os.path.join(
    MARKUP_DATA_DIR,
    "rt_pa_all_years_summary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EDCs
# ============================================================

TARGET_EDCS = [
    "APS",
    "DUQ",
    "METED",
    "PECO",
    "PENELEC",
    "PPL",
    "AECO",
    "BGE",
    "DPL",
    "JCPL",
    "PEPCO"
]

PA_EDCS = [
    "APS",
    "DUQ",
    "METED",
    "PECO",
    "PENELEC",
    "PPL"
]


# ============================================================
# Normalize EDC
# ============================================================

def normalize_edc(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if "west penn" in text:
        return "APS"

    if text == "aps":
        return "APS"

    if "duquesne" in text:
        return "DUQ"

    if "met-ed" in text or "meted" in text:
        return "METED"

    if "peco" in text:
        return "PECO"

    if "penelec" in text:
        return "PENELEC"

    if "ppl" in text:
        return "PPL"

    if (
        "atlantic city electric" in text
        or "aeco" in text
    ):
        return "AECO"

    if (
        "bge" in text
        or "baltimore gas" in text
    ):
        return "BGE"

    if (
        "delmarva" in text
        or text == "dpl"
    ):
        return "DPL"

    if (
        "jersey central" in text
        or "jcpl" in text
    ):
        return "JCPL"

    if (
        "pepco" in text
        or "potomac electric" in text
    ):
        return "PEPCO"

    return str(value).strip().upper()


# ============================================================
# Parsing helpers
# ============================================================

def number_from_text(value):
    if value is None:
        return None

    if isinstance(value, bool):
        if value is False:
            return 0.0
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).strip()

    if text == "":
        return None

    if text.lower() in [
        "no",
        "none",
        "false",
        "no fee",
        "n/a",
        "na"
    ]:
        return 0.0

    match = re.search(
        r"\$?\s*([0-9]+(?:\.[0-9]+)?)",
        text
    )

    if match:
        return float(
            match.group(1)
        )

    return None


def parse_rate(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    if isinstance(value, list):
        if len(value) == 0:
            return None

        first = value[0]

        if isinstance(first, dict):
            return number_from_text(
                first.get("amount")
            )

        return number_from_text(
            first
        )

    if isinstance(value, dict):
        return number_from_text(
            value.get("amount")
        )

    return number_from_text(
        value
    )


def parse_created_at(value):
    if isinstance(value, dict):
        year = value.get("year")
        month = value.get("month")

        if (
            year is not None
            and month is not None
        ):
            return (
                int(year),
                int(month)
            )

    if value is not None:
        try:
            date = pd.to_datetime(
                value
            )

            return (
                int(date.year),
                int(date.month)
            )

        except Exception:
            pass

    return None, None


# ============================================================
# WattBuy parsing
# ============================================================

def parse_signup_fee(record):
    original = record.get(
        "original_data",
        {}
    )

    values = []

    keys = [
        "enrollment_fee",
        "enrollmentFee",
        "enroll_fee",
        "Enrollment Fee"
    ]

    for key in keys:
        if key in record:
            values.append(
                record.get(key)
            )

        if (
            isinstance(original, dict)
            and key in original
        ):
            values.append(
                original.get(key)
            )

    for value in values:
        if value is None:
            continue

        if isinstance(value, bool):
            if value is False:
                return 0.0
            continue

        text = str(value).strip()

        if text == "":
            continue

        if text.startswith("http"):
            continue

        if text.lower() in [
            "no",
            "false",
            "none",
            "no fee",
            "$0",
            "$0.00",
            "0",
            "0.0",
            "0.00"
        ]:
            return 0.0

        match = re.search(
            r"\$?\s*([0-9]+(?:\.[0-9]+)?)",
            text
        )

        if match:
            fee = float(
                match.group(1)
            )

            # WattBuy enrollment_fee is stored in cents.
            # Convert it to dollars for B.b/B.e.
            return fee / 100.0

    return None

def parse_early_termination_fee(record):
    original = record.get(
        "original_data",
        {}
    )

    if isinstance(original, dict):
        possible_keys = [
            "Cancellation Fee",
            "Early Termination Fee",
            "Early Termination"
        ]

        for key in possible_keys:
            if key not in original:
                continue

            raw = original.get(
                key
            )

            if raw is None:
                continue

            text = str(raw).strip()

            if text.lower() in [
                "",
                "no",
                "false",
                "none",
                "no fee",
                "$0",
                "$0.00",
                "0",
                "0.0",
                "0.00"
            ]:
                return 0.0

            value = number_from_text(
                text
            )

            if value is not None:
                return value

    value = record.get(
        "early_term_fee"
    )

    if value is None:
        return None

    value = number_from_text(
        value
    )

    if value is None:
        return None

    if value > 500:
        return value / 100.0

    return value


def parse_green_percentage(record):
    value = record.get(
        "green_percentage"
    )

    if (
        value is not None
        and not pd.isna(value)
    ):
        return number_from_text(
            value
        )

    original = record.get(
        "original_data",
        {}
    )

    if isinstance(original, dict):
        possible_keys = [
            "Renewable Energy",
            "Renewable",
            "Green Percentage"
        ]

        for key in possible_keys:
            if key not in original:
                continue

            raw = original.get(
                key
            )

            if raw is None:
                continue

            text = str(raw).strip()

            match = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*%",
                text
            )

            if match:
                return float(
                    match.group(1)
                )

            if text.lower() in [
                "",
                "no",
                "false",
                "none",
                "0",
                "0.0",
                "0%"
            ]:
                return 0.0

    is_green = record.get(
        "is_green"
    )

    if is_green is False:
        return 0.0

    return None


def parse_term(record):
    value = record.get(
        "term"
    )

    if value is not None:
        number = number_from_text(
            value
        )

        if number is not None:
            return number

    original = record.get(
        "original_data",
        {}
    )

    if isinstance(original, dict):
        possible_keys = [
            "Term",
            "Term Length",
            "Contract Term"
        ]

        for key in possible_keys:
            if key not in original:
                continue

            value = number_from_text(
                original.get(key)
            )

            if value is not None:
                return value

    return None


def walk_json(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from walk_json(
                item
            )

    elif isinstance(obj, dict):
        keys = set(
            obj.keys()
        )

        looks_like_plan = (
            (
                "supplier_name" in keys
                or "supplier" in keys
            )
            and
            (
                "rate" in keys
                or "original_data" in keys
            )
        )

        if looks_like_plan:
            yield obj

        for value in obj.values():
            if isinstance(
                value,
                (list, dict)
            ):
                yield from walk_json(
                    value
                )


# ============================================================
# Load WattBuy
# ============================================================

def load_wattbuy_raw():
    raw_dir = os.path.join(
        WATTBUY_DIR,
        "data",
        "raw",
        "zip-data"
    )

    files = [
        os.path.join(
            raw_dir,
            "pennsylvania_plans_2019.json.zip"
        ),
        os.path.join(
            raw_dir,
            "pennsylvania_plans_2020.json.zip"
        ),
        os.path.join(
            raw_dir,
            "pennsylvania_plans_2021.json.zip"
        ),
        os.path.join(
            raw_dir,
            "pennsylvania_plans_2022.json.zip"
        )
    ]

    rows = []

    for file in files:
        if not os.path.exists(file):
            print(
                "WARNING: missing WattBuy raw file:",
                file
            )
            continue

        print(
            "Reading",
            os.path.basename(file)
        )

        with zipfile.ZipFile(
            file,
            "r"
        ) as z:
            json_files = [
                name
                for name in z.namelist()
                if name.endswith(".json")
                and not name.startswith(
                    "__MACOSX"
                )
            ]

            for json_name in json_files:
                try:
                    with z.open(
                        json_name
                    ) as f:
                        data = json.load(
                            f
                        )

                except Exception as e:
                    print(
                        "Could not read",
                        json_name,
                        ":",
                        e
                    )
                    continue

                for record in walk_json(
                    data
                ):
                    year, month = (
                        parse_created_at(
                            record.get(
                                "created_at"
                            )
                        )
                    )

                    if (
                        year is None
                        or month is None
                    ):
                        continue

                    utility = (
                        record.get(
                            "utility_name"
                        )
                        or record.get(
                            "utility"
                        )
                    )

                    supplier = (
                        record.get(
                            "supplier_name"
                        )
                        or record.get(
                            "supplier"
                        )
                    )

                    edc = normalize_edc(
                        utility
                    )

                    if edc not in PA_EDCS:
                        continue

                    rate = parse_rate(
                        record.get(
                            "rate"
                        )
                    )

                    term = parse_term(
                        record
                    )

                    signup_fee = (
                        parse_signup_fee(
                            record
                        )
                    )

                    early_fee = (
                        parse_early_termination_fee(
                            record
                        )
                    )

                    green_percentage = (
                        parse_green_percentage(
                            record
                        )
                    )

                    rows.append(
                        {
                            "source":
                                "WattBuy",
                            "Year":
                                year,
                            "Month":
                                month,
                            "EDC":
                                edc,
                            "EGS":
                                supplier,
                            "rate":
                                rate,
                            "term_months":
                                term,
                            "signup_fee":
                                signup_fee,
                            "early_termination_fee":
                                early_fee,
                            "green_percentage":
                                green_percentage
                        }
                    )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise ValueError(
            "No WattBuy raw records were extracted."
        )

    df["date"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=1
        )
    )

    return df


# ============================================================
# PA Switch
# ============================================================

def parse_pa_switch_signup(value):
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        if value is False:
            return 0.0
        return None

    text = str(value).strip()

    if text == "":
        return None

    if text.startswith(
        "http"
    ):
        return None

    if text.lower() in [
        "false",
        "no",
        "none",
        "no fee",
        "0",
        "0.0",
        "0.00"
    ]:
        return 0.0

    return number_from_text(
        text
    )


def load_pa_switch_signup():
    files = sorted(
        glob.glob(
            os.path.join(
                PA_SWITCH_DIR,
                "data",
                "monthly",
                "power",
                "*.csv"
            )
        )
    )

    frames = []

    for file in files:
        df = pd.read_csv(
            file,
            low_memory=False
        )

        required = [
            "egs",
            "edc",
            "rate",
            "term",
            "green_percentage",
            "cancel_fee",
            "enroll_fee",
            "snapshot_month"
        ]

        if not all(
            col in df.columns
            for col in required
        ):
            continue

        temp = pd.DataFrame()

        temp["source"] = (
            "PA Switch"
        )

        temp["EGS"] = (
            df["egs"]
        )

        temp["EDC"] = (
            df["edc"]
            .apply(
                normalize_edc
            )
        )

        temp["rate"] = (
            pd.to_numeric(
                df["rate"],
                errors="coerce"
            )
            * 100.0
        )

        temp["term_months"] = (
            pd.to_numeric(
                df["term"],
                errors="coerce"
            )
        )

        temp["green_percentage"] = (
            pd.to_numeric(
                df[
                    "green_percentage"
                ],
                errors="coerce"
            )
        )

        temp[
            "early_termination_fee"
        ] = pd.to_numeric(
            df["cancel_fee"],
            errors="coerce"
        )

        temp["signup_fee"] = (
            df["enroll_fee"]
            .apply(
                parse_pa_switch_signup
            )
        )

        dates = pd.to_datetime(
            df["snapshot_month"],
            format="%Y-%m",
            errors="coerce"
        )

        temp["Year"] = (
            dates.dt.year
        )

        temp["Month"] = (
            dates.dt.month
        )

        temp["date"] = (
            dates
        )

        temp = temp[
            temp["EDC"].isin(
                PA_EDCS
            )
        ].copy()

        temp = temp[
            temp[
                "signup_fee"
            ].notna()
        ].copy()

        frames.append(
            temp
        )

    if len(frames) == 0:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# Load PTC
# ============================================================

def load_ptc():
    df = pd.read_csv(
        MARKUP_FILE,
        low_memory=False
    )

    required = [
        "Year",
        "Month",
        "EDC",
        "PTC"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in markup_full_real.csv: "
            + str(missing)
        )

    df["EDC"] = (
        df["EDC"]
        .apply(
            normalize_edc
        )
    )

    df["PTC"] = (
        pd.to_numeric(
            df["PTC"],
            errors="coerce"
        )
    )

    df = df[
        df["EDC"].isin(
            PA_EDCS
        )
    ].copy()

    ptc = (
        df[
            [
                "Year",
                "Month",
                "EDC",
                "PTC"
            ]
        ]
        .dropna(
            subset=[
                "PTC"
            ]
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
            PTC=(
                "PTC",
                "first"
            )
        )
    )

    return ptc


# ============================================================
# Load PJM RT
# ============================================================

def load_rt():
    df = pd.read_csv(
        RT_FILE
    )

    required = [
        "Year",
        "Month",
        "EDC",
        "Average",
        "Median"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in RT file: "
            + str(missing)
        )

    df["EDC"] = (
        df["EDC"]
        .apply(
            normalize_edc
        )
    )

    df = df[
        df["EDC"].isin(
            PA_EDCS
        )
    ].copy()

    df["Average"] = (
        pd.to_numeric(
            df["Average"],
            errors="coerce"
        )
    )

    df["Median"] = (
        pd.to_numeric(
            df["Median"],
            errors="coerce"
        )
    )

    # PJM LMP is $/MWh.
    # Convert to cents/kWh.
    df["PJM_RT_Average"] = (
        df["Average"] / 10.0
    )

    df["PJM_RT_Median"] = (
        df["Median"] / 10.0
    )

    return df[
        [
            "Year",
            "Month",
            "EDC",
            "PJM_RT_Average",
            "PJM_RT_Median"
        ]
    ]


# ============================================================
# B.b
# Average and median signup fee
# ============================================================

def create_b_b(offers):
    print()
    print("=" * 65)
    print("Creating B.b")
    print("=" * 65)

    b = offers[
        offers[
            "signup_fee"
        ].notna()
    ].copy()

    b = b[
        b["EDC"].isin(
            PA_EDCS
        )
    ].copy()

    if b.empty:
        print(
            "No reliable signup-fee observations found."
        )
        return pd.DataFrame()

    monthly = (
        b.groupby(
            [
                "Year",
                "Month",
                "EDC"
            ],
            as_index=False
        )
        .agg(
            Average_Signup_Fee=(
                "signup_fee",
                "mean"
            ),
            Median_Signup_Fee=(
                "signup_fee",
                "median"
            ),
            Number_of_Offers=(
                "signup_fee",
                "size"
            ),
            Number_of_EGS=(
                "EGS",
                "nunique"
            )
        )
    )

    monthly["date"] = (
        pd.to_datetime(
            dict(
                year=monthly[
                    "Year"
                ],
                month=monthly[
                    "Month"
                ],
                day=1
            )
        )
    )

    monthly = monthly.sort_values(
        [
            "EDC",
            "date"
        ]
    )

    monthly.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "B_b_signup_fee_data.csv"
        ),
        index=False
    )

    for edc in PA_EDCS:
        temp = monthly[
            monthly["EDC"]
            == edc
        ].sort_values(
            "date"
        )

        if temp.empty:
            print(
                "B.b:",
                edc,
                "- no reliable signup fee data"
            )
            continue

        plt.figure(
            figsize=(
                11,
                6
            )
        )

        plt.plot(
            temp["date"],
            temp[
                "Average_Signup_Fee"
            ],
            marker="o",
            markersize=3,
            label="Average Signup Fee"
        )

        plt.plot(
            temp["date"],
            temp[
                "Median_Signup_Fee"
            ],
            marker="o",
            markersize=3,
            label="Median Signup Fee"
        )

        plt.title(
            edc
            + " - Average and Median Signup Fee"
        )

        plt.xlabel(
            "Time"
        )

        plt.ylabel(
            "Signup Fee ($)"
        )

        plt.legend()

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        filename = (
            edc
            + "_signup_fee_average_median.png"
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                filename
            ),
            dpi=300
        )

        plt.close()

        print(
            "B.b:",
            edc,
            "| months:",
            temp[
                "date"
            ].nunique(),
            "| observations:",
            int(
                temp[
                    "Number_of_Offers"
                ].sum()
            )
        )

    return monthly


# ============================================================
# B.e
# PTC-like offers:
# 6 months
# no early termination fee
# no renewables
# no signup fee
# ============================================================

def create_b_e(
    offers,
    ptc,
    rt
):
    print()
    print("=" * 65)
    print("Creating B.e")
    print("=" * 65)

    offers = offers[
        offers["EDC"].isin(
            PA_EDCS
        )
    ].copy()

    # All characteristics must be explicitly known.
    known = offers[
        offers[
            "term_months"
        ].notna()
        & offers[
            "early_termination_fee"
        ].notna()
        & offers[
            "green_percentage"
        ].notna()
        & offers[
            "signup_fee"
        ].notna()
        & offers[
            "rate"
        ].notna()
    ].copy()

    print()
    print(
        "Known-characteristic offers:",
        len(known)
    )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    print()
    print("B.e FILTER CHECK")

    for edc in PA_EDCS:
        x = known[
            known["EDC"] == edc
        ]

        six_month = (
            x["term_months"] == 6
        )

        no_termination = (
            x[
                "early_termination_fee"
            ] == 0
        )

        no_renewables = (
            x[
                "green_percentage"
            ] == 0
        )

        no_signup = (
            x[
                "signup_fee"
            ] == 0
        )

        all_four = (
            six_month
            & no_termination
            & no_renewables
            & no_signup
        )

        print()
        print(edc)

        print(
            "Known rows:",
            len(x)
        )

        print(
            "6-month:",
            int(
                six_month.sum()
            )
        )

        print(
            "No termination fee:",
            int(
                no_termination.sum()
            )
        )

        print(
            "No renewables:",
            int(
                no_renewables.sum()
            )
        )

        print(
            "No signup fee:",
            int(
                no_signup.sum()
            )
        )

        print(
            "All four:",
            int(
                all_four.sum()
            )
        )

    # --------------------------------------------------------
    # Strict PTC-like filter
    # --------------------------------------------------------

    similar = known[
        (
            known[
                "term_months"
            ] == 6
        )
        & (
            known[
                "early_termination_fee"
            ] == 0
        )
        & (
            known[
                "green_percentage"
            ] == 0
        )
        & (
            known[
                "signup_fee"
            ] == 0
        )
    ].copy()

    print()
    print(
        "PTC-like offers before PTC merge:",
        len(similar)
    )

    # --------------------------------------------------------
    # Merge PTC
    # --------------------------------------------------------

    similar = similar.merge(
        ptc,
        on=[
            "Year",
            "Month",
            "EDC"
        ],
        how="left"
    )

    print(
        "PTC-like offers with PTC:",
        int(
            similar[
                "PTC"
            ].notna().sum()
        )
    )

    print(
        "PTC-like offers missing PTC:",
        int(
            similar[
                "PTC"
            ].isna().sum()
        )
    )

    # --------------------------------------------------------
    # Missing PTC diagnostic
    # --------------------------------------------------------

    print()
    print(
        "Missing PTC by EDC:"
    )

    missing_ptc = similar[
        similar[
            "PTC"
        ].isna()
    ].copy()

    if missing_ptc.empty:
        print(
            "None"
        )

    else:
        print(
            missing_ptc
            .groupby(
                "EDC"
            )
            .size()
        )

    print()
    print(
        "PECO PTC-like offers missing PTC:"
    )

    peco_missing = similar[
        (
            similar["EDC"]
            == "PECO"
        )
        & (
            similar[
                "PTC"
            ].isna()
        )
    ].copy()

    if peco_missing.empty:
        print(
            "None"
        )

    else:
        print(
            peco_missing[
                [
                    "Year",
                    "Month",
                    "EGS",
                    "rate",
                    "term_months",
                    "early_termination_fee",
                    "green_percentage",
                    "signup_fee"
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Keep only matched PTC
    # --------------------------------------------------------

    similar = similar[
        similar[
            "PTC"
        ].notna()
    ].copy()

    if similar.empty:
        print(
            "No PTC-like offers have matching PTC data."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Above / below PTC
    # --------------------------------------------------------

    similar[
        "comparison"
    ] = "Equal"

    similar.loc[
        similar["rate"]
        > similar["PTC"],
        "comparison"
    ] = "Above"

    similar.loc[
        similar["rate"]
        < similar["PTC"],
        "comparison"
    ] = "Below"

    # --------------------------------------------------------
    # Save offer-level data
    # --------------------------------------------------------

    similar = similar.sort_values(
        [
            "EDC",
            "Year",
            "Month",
            "EGS"
        ]
    )

    similar.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "B_e_ptc_like_offers.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Monthly counts
    # --------------------------------------------------------

    counts = (
        similar.groupby(
            [
                "Year",
                "Month",
                "EDC",
                "comparison"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    for col in [
        "Above",
        "Below",
        "Equal"
    ]:
        if col not in counts.columns:
            counts[col] = 0

    counts["Total"] = (
        counts["Above"]
        + counts["Below"]
        + counts["Equal"]
    )

    counts[
        "Share_Above_PTC"
    ] = (
        100.0
        * counts["Above"]
        / counts["Total"]
    )

    counts[
        "Share_Below_PTC"
    ] = (
        100.0
        * counts["Below"]
        / counts["Total"]
    )

    counts[
        "Share_Equal_PTC"
    ] = (
        100.0
        * counts["Equal"]
        / counts["Total"]
    )

    # --------------------------------------------------------
    # Merge PJM RT
    # --------------------------------------------------------

    counts = counts.merge(
        rt,
        on=[
            "Year",
            "Month",
            "EDC"
        ],
        how="left"
    )

    counts[
        "date"
    ] = pd.to_datetime(
        dict(
            year=counts[
                "Year"
            ],
            month=counts[
                "Month"
            ],
            day=1
        )
    )

    counts = counts.sort_values(
        [
            "EDC",
            "date"
        ]
    )

    counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "B_e_ptc_like_above_below_data.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    for edc in PA_EDCS:
        temp = counts[
            counts["EDC"]
            == edc
        ].sort_values(
            "date"
        )

        if temp.empty:
            print(
                "B.e:",
                edc,
                "- no PTC-like data"
            )
            continue

        fig, ax1 = plt.subplots(
            figsize=(
                11,
                6
            )
        )

        ax1.plot(
            temp["date"],
            temp[
                "Share_Above_PTC"
            ],
            marker="o",
            markersize=3,
            label="Offers Above PTC (%)"
        )

        ax1.plot(
            temp["date"],
            temp[
                "Share_Below_PTC"
            ],
            marker="o",
            markersize=3,
            label="Offers Below PTC (%)"
        )

        ax1.set_xlabel(
            "Time"
        )

        ax1.set_ylabel(
            "Share of PTC-like Offers (%)"
        )

        ax1.set_ylim(
            0,
            100
        )

        ax1.grid(
            alpha=0.25
        )

        ax2 = ax1.twinx()

        ax2.plot(
            temp["date"],
            temp[
                "PJM_RT_Average"
            ],
            linestyle="--",
            label="PJM RT Average"
        )

        ax2.set_ylabel(
            "PJM RT Price (cents/kWh)"
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
            edc
            + " - PTC-like Offers Above/Below PTC with PJM RT"
        )

        fig.tight_layout()

        filename = (
            edc
            + "_ptc_like_above_below_ptc_rt.png"
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                filename
            ),
            dpi=300
        )

        plt.close()

        print(
            "B.e:",
            edc,
            "| months:",
            temp[
                "date"
            ].nunique(),
            "| offers:",
            int(
                temp[
                    "Total"
                ].sum()
            ),
            "| RT missing:",
            int(
                temp[
                    "PJM_RT_Average"
                ].isna().sum()
            )
        )

    return counts


# ============================================================
# Final coverage
# ============================================================

def print_coverage(
    offers,
    b_b,
    b_e
):
    print()
    print("=" * 65)
    print("FINAL COVERAGE")
    print("=" * 65)

    print()
    print(
        "Reliable signup-fee observations:"
    )

    for edc in PA_EDCS:
        n = len(
            offers[
                (
                    offers["EDC"]
                    == edc
                )
                & (
                    offers[
                        "signup_fee"
                    ].notna()
                )
            ]
        )

        print(
            edc,
            ":",
            n
        )

    print()
    print(
        "B.b EDCs created:"
    )

    if not b_b.empty:
        print(
            sorted(
                b_b[
                    "EDC"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )
    else:
        print(
            []
        )

    print()
    print(
        "B.e EDCs created:"
    )

    if not b_e.empty:
        print(
            sorted(
                b_e[
                    "EDC"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )
    else:
        print(
            []
        )

    print()
    print(
        "Other 11-EDC targets not supported by"
    )

    print(
        "the Pennsylvania retail offer sources:"
    )

    print(
        [
            edc
            for edc in TARGET_EDCS
            if edc not in PA_EDCS
        ]
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print(
        "Presentation Offer Analysis"
    )
    print(
        "Professor requirements: B.b and B.e"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # WattBuy
    # --------------------------------------------------------

    print()
    print(
        "Loading WattBuy raw data..."
    )

    wattbuy = (
        load_wattbuy_raw()
    )

    print(
        "WattBuy rows:",
        len(wattbuy)
    )

    # --------------------------------------------------------
    # PA Switch
    # --------------------------------------------------------

    print()
    print(
        "Loading reliable newer PA Switch signup data..."
    )

    pa_switch = (
        load_pa_switch_signup()
    )

    if not pa_switch.empty:
        print(
            "PA Switch reliable rows:",
            len(pa_switch)
        )

        offers = pd.concat(
            [
                wattbuy,
                pa_switch
            ],
            ignore_index=True
        )

    else:
        print(
            "No additional reliable PA Switch signup rows."
        )

        offers = (
            wattbuy.copy()
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    offers = (
        offers.drop_duplicates(
            subset=[
                "Year",
                "Month",
                "EDC",
                "EGS",
                "rate",
                "term_months",
                "signup_fee",
                "early_termination_fee",
                "green_percentage"
            ]
        )
        .copy()
    )

    offers = offers[
        offers[
            "EDC"
        ].isin(
            PA_EDCS
        )
    ].copy()

    print()
    print(
        "Final offer rows:",
        len(offers)
    )

    # --------------------------------------------------------
    # PTC
    # --------------------------------------------------------

    print()
    print(
        "Loading PTC..."
    )

    ptc = (
        load_ptc()
    )

    print(
        "PTC rows:",
        len(ptc)
    )

    # --------------------------------------------------------
    # PJM RT
    # --------------------------------------------------------

    print()
    print(
        "Loading PJM RT..."
    )

    rt = (
        load_rt()
    )

    print(
        "RT rows:",
        len(rt)
    )

    # --------------------------------------------------------
    # B.b
    # --------------------------------------------------------

    b_b = create_b_b(
        offers
    )

    # --------------------------------------------------------
    # B.e
    # --------------------------------------------------------

    b_e = create_b_e(
        offers,
        ptc,
        rt
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    print_coverage(
        offers,
        b_b,
        b_e
    )

    print()
    print("=" * 65)
    print("DONE")
    print("=" * 65)

    print(
        "Outputs saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()