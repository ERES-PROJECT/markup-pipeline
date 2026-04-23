import os
import re
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLANS_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "plans")
PTC_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "ptc")

OUT_RETAIL = os.path.join(BASE_DIR, "retail_from_plans_clean.csv")
OUT_MARKUP = os.path.join(BASE_DIR, "markup_full.csv")
OUT_MOMENTS = os.path.join(BASE_DIR, "markup_moments.csv")

START_YEAR = 2017
START_MONTH = 8
END_YEAR = 2025
END_MONTH = 4


def in_range(year, month):
    return (
        (year > START_YEAR or (year == START_YEAR and month >= START_MONTH))
        and
        (year < END_YEAR or (year == END_YEAR and month <= END_MONTH))
    )


def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def normalize_upper(x):
    x = normalize_text(x)
    return x.upper() if x is not None else None


def extract_year_month_from_filename(filename):
    m = re.search(r"(20\d{2})-(\d{2})", filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def month_to_int(x):
    if pd.isna(x):
        return None

    if isinstance(x, (int, float)):
        return int(x)

    s = str(x).strip()
    month_map = {
        "JANUARY": 1,
        "FEBRUARY": 2,
        "MARCH": 3,
        "APRIL": 4,
        "MAY": 5,
        "JUNE": 6,
        "JULY": 7,
        "AUGUST": 8,
        "SEPTEMBER": 9,
        "OCTOBER": 10,
        "NOVEMBER": 11,
        "DECEMBER": 12,
    }

    if s.upper() in month_map:
        return month_map[s.upper()]

    try:
        return int(float(s))
    except Exception:
        return None


def find_col(df, candidates):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    return None


# -----------------------------
# 你的 mapping 表 -> final zone
# -----------------------------
def map_final_zone(x):
    x = normalize_upper(x)
    if x is None:
        return None

    # short retail codes / existing parsed values
    direct_map = {
        "CIT": "PPL",
        "DQE": "DUQ",
        "MET": "METED",
        "PEC": "PECO",
        "PEN": "PENELEC",
        "PEP": "APS",
        "PIK": None,         # exclude
        "PPL": "PPL",
        "UGI": "PPL",
        "WEL": "PENELEC",
        "WES": "APS",
        "DUQ": "DUQ",
        "METED": "METED",
        "PECO": "PECO",
        "PENELEC": "PENELEC",
        "APS": "APS",
    }
    if x in direct_map:
        return direct_map[x]

    # long names from plans / ptc
    if "DUQUESNE" in x:
        return "DUQ"
    if "MET ED" in x or "MET-ED" in x:
        return "METED"
    if "PECO" in x:
        return "PECO"
    if "PENELEC" in x:
        return "PENELEC"
    if "PENN POWER" in x or "PENNSYLVANIA POWER" in x:
        return "APS"
    if "WEST PENN" in x or "ALLEGHENY" in x:
        return "APS"
    if "PIKE COUNTY" in x:
        return None
    if "PPL" in x or "CITIZENS" in x or x == "UGI":
        return "PPL"
    if "WELLSBORO" in x:
        return "PENELEC"

    return None


def load_plans():
    files = sorted(glob.glob(os.path.join(PLANS_DIR, "*.csv")))
    dfs = []

    if not files:
        raise ValueError(f"No files found in plans dir: {PLANS_DIR}")

    for f in files:
        df = pd.read_csv(f)
        filename = os.path.basename(f)

        year_from_name, month_from_name = extract_year_month_from_filename(filename)

        company_col = find_col(df, ["company", "supplier", "company_name", "supplier_name"])
        rate_col = find_col(df, ["rate", "price", "price_per_kwh", "cents_per_kwh"])
        edc_col = find_col(df, ["edc", "utility", "territory"])
        year_col = find_col(df, ["year"])
        month_col = find_col(df, ["month"])

        if company_col is None or rate_col is None or edc_col is None:
            continue

        temp = pd.DataFrame()
        temp["Year"] = pd.to_numeric(df[year_col], errors="coerce") if year_col else year_from_name
        temp["Month"] = df[month_col].apply(month_to_int) if month_col else month_from_name
        temp["supplier"] = df[company_col].apply(normalize_text)
        temp["price"] = pd.to_numeric(df[rate_col], errors="coerce")
        temp["EDC"] = df[edc_col].apply(map_final_zone)

        temp = temp.dropna(subset=["Year", "Month", "supplier", "price", "EDC"]).copy()
        temp["Year"] = temp["Year"].astype(int)
        temp["Month"] = temp["Month"].astype(int)

        temp = temp[temp.apply(lambda r: in_range(r["Year"], r["Month"]), axis=1)].copy()

        # 合理价格范围，过滤脏值
        temp = temp[(temp["price"] >= 4) & (temp["price"] <= 25)].copy()

        # 去重
        temp = temp.drop_duplicates(subset=["Year", "Month", "EDC", "supplier", "price"])

        dfs.append(temp)

    if not dfs:
        raise ValueError("No usable plans csv files found. Check monthly/plans column names.")

    retail = pd.concat(dfs, ignore_index=True)
    retail = retail.sort_values(["Year", "Month", "EDC", "supplier", "price"]).reset_index(drop=True)
    retail.to_csv(OUT_RETAIL, index=False)
    return retail


def load_ptc():
    files = sorted(glob.glob(os.path.join(PTC_DIR, "*.csv")))
    dfs = []

    if not files:
        raise ValueError(f"No files found in ptc dir: {PTC_DIR}")

    for f in files:
        df = pd.read_csv(f)
        filename = os.path.basename(f)

        year_from_name, month_from_name = extract_year_month_from_filename(filename)

        edc_col = find_col(df, ["edc"])
        rate_col = find_col(df, ["rate"])
        year_col = find_col(df, ["year"])
        month_col = find_col(df, ["month"])

        if edc_col is None or rate_col is None:
            continue

        temp = pd.DataFrame()
        temp["Year"] = pd.to_numeric(df[year_col], errors="coerce") if year_col else year_from_name
        temp["Month"] = df[month_col].apply(month_to_int) if month_col else month_from_name
        temp["EDC"] = df[edc_col].apply(map_final_zone)
        temp["PTC"] = pd.to_numeric(df[rate_col], errors="coerce")

        temp = temp.dropna(subset=["Year", "Month", "EDC", "PTC"]).copy()
        temp["Year"] = temp["Year"].astype(int)
        temp["Month"] = temp["Month"].astype(int)

        temp = temp[temp.apply(lambda r: in_range(r["Year"], r["Month"]), axis=1)].copy()

        # 同一月同一zone若有多行，取平均
        temp = temp.groupby(["Year", "Month", "EDC"], as_index=False)["PTC"].mean()

        dfs.append(temp)

    if not dfs:
        raise ValueError("No usable ptc csv files found.")

    return pd.concat(dfs, ignore_index=True)


def build_markup():
    retail = load_plans()
    ptc = load_ptc()

    df = retail.merge(ptc, on=["Year", "Month", "EDC"], how="left")

    print("Retail rows before merge:", len(retail))
    print("Rows missing PTC after merge:", int(df["PTC"].isna().sum()))

    df = df.dropna(subset=["PTC"]).copy()
    df["markup"] = df["price"] - df["PTC"]

    df = df.sort_values(["Year", "Month", "EDC", "supplier", "price"]).reset_index(drop=True)
    df.to_csv(OUT_MARKUP, index=False)
    return df


def compute_moments(df):
    def summarize(g):
        d = g["markup"].dropna()
        return pd.Series({
            "n_offers": len(d),
            "MeanMarkup": d.mean(),
            "Variance": d.var(),
            "ShareAbovePTC": (d > 0).mean(),
            "Q25": d.quantile(0.25),
            "Q50": d.quantile(0.50),
            "Q75": d.quantile(0.75),
        })

    moments = (
        df.groupby(["Year", "Month", "EDC"])
        .apply(summarize)
        .reset_index()
        .sort_values(["Year", "Month", "EDC"])
        .reset_index(drop=True)
    )

    moments.to_csv(OUT_MOMENTS, index=False)
    return moments


if __name__ == "__main__":
    markup_df = build_markup()
    moments_df = compute_moments(markup_df)

    print("\nDone.")
    print("\nmarkup_full preview:")
    print(markup_df.head(10))

    print("\nmarkup_moments preview:")
    print(moments_df.head(20))

    print("\nSaved files:")
    print(OUT_RETAIL)
    print(OUT_MARKUP)
    print(OUT_MOMENTS)