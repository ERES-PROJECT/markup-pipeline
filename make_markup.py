import pandas as pd
import glob
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OCA_PTC_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "ptc")
DA_FILE = os.path.join(BASE_DIR, "..", "pjm-pipeline", "data", "daily", "da_pa_all_years_summary.csv")
RT_FILE = os.path.join(BASE_DIR, "..", "pjm-pipeline", "data", "daily", "rt_pa_all_years_summary.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "markup_2017_01_to_2024_08.csv")

START_YEAR = 2017
START_MONTH = 1
END_YEAR = 2024
END_MONTH = 8

# OCA edc name -> PJM edc name
EDC_MAP = {
    "DUQUESNE LIGHT": "DUQ",
    "MET ED": "METED",
    "MET ED 1": "METED",
    "MET-ED": "METED",
    "PECO ENERGY": "PECO",
    "PPL ELECTRIC UTILITIES": "PPL",
    "PENELEC": "PENELEC",
    "PENNELEC": "PENELEC",
    "PENN POWER": "PENELEC",
    "WEST PENN POWER": "APS",
    "PIKE COUNTY LIGHT AND POWER": None
}


def in_range(year, month):
    return (year > START_YEAR or (year == START_YEAR and month >= START_MONTH)) and \
           (year < END_YEAR or (year == END_YEAR and month <= END_MONTH))


def normalize_text(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()


def map_oca_edc(edc_value):
    x = normalize_text(edc_value)
    if pd.isna(x):
        return None

    if x in EDC_MAP:
        return EDC_MAP[x]

    if "DUQUESNE" in x:
        return "DUQ"
    if "MET ED" in x or "MET-ED" in x:
        return "METED"
    if "PECO" in x:
        return "PECO"
    if "PPL" in x:
        return "PPL"
    if "PENELEC" in x:
        return "PENELEC"
    if "PENN POWER" in x:
        return "PENELEC"
    if "WEST PENN" in x:
        return "APS"
    if "PIKE COUNTY" in x:
        return None

    return None


def load_oca_ptc(ptc_dir):
    all_files = sorted(glob.glob(os.path.join(ptc_dir, "oca_ptc_*.csv")))
    dfs = []

    for file in all_files:
        filename = os.path.basename(file)

        m = re.search(r"oca_ptc_(\d{4})-(\d{2})\.csv", filename)
        if not m:
            continue

        year = int(m.group(1))
        month_num = int(m.group(2))

        if not in_range(year, month_num):
            continue

        df = pd.read_csv(file)

        required = {"edc", "rate"}
        if not required.issubset(df.columns):
            print(f"Skipping {filename}: missing columns {required}")
            print("Columns found:", list(df.columns))
            continue

        temp = df[["edc", "rate"]].copy()
        temp.columns = ["EDC_raw", "PTC"]

        temp["Year"] = year
        temp["Month"] = month_num
        temp["EDC"] = temp["EDC_raw"].apply(map_oca_edc)

        temp["PTC"] = (
            temp["PTC"]
            .astype(str)
            .str.extract(r"([-]?\d+\.?\d*)", expand=False)
        )
        temp["PTC"] = pd.to_numeric(temp["PTC"], errors="coerce")

        temp = temp.dropna(subset=["EDC", "PTC"])
        temp = temp.groupby(["Year", "Month", "EDC"], as_index=False)["PTC"].mean()

        dfs.append(temp)

    if not dfs:
        raise ValueError("No valid OCA PTC files were loaded.")

    ptc_df = pd.concat(dfs, ignore_index=True)
    return ptc_df[["Year", "Month", "EDC", "PTC"]]


def load_da(da_file):
    df = pd.read_csv(da_file)

    required_cols = {"Year", "Month", "EDC", "Average"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DA file must contain columns: {required_cols}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["EDC"] = df["EDC"].apply(normalize_text)
    df["DA_raw"] = pd.to_numeric(df["Average"], errors="coerce")

    df = df.dropna(subset=["Year", "Month"])
    df = df[df.apply(lambda r: in_range(int(r["Year"]), int(r["Month"])), axis=1)].copy()

    # $/MWh -> cents/kWh
    df["DA"] = df["DA_raw"] / 10.0

    df = df[["Year", "Month", "EDC", "DA"]].dropna()
    return df


def load_rt(rt_file):
    df = pd.read_csv(rt_file)

    required_cols = {"Year", "Month", "EDC", "Average"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"RT file must contain columns: {required_cols}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["EDC"] = df["EDC"].apply(normalize_text)
    df["RT_raw"] = pd.to_numeric(df["Average"], errors="coerce")

    df = df.dropna(subset=["Year", "Month"])
    df = df[df.apply(lambda r: in_range(int(r["Year"]), int(r["Month"])), axis=1)].copy()

    # $/MWh -> cents/kWh
    df["RT"] = df["RT_raw"] / 10.0

    df = df[["Year", "Month", "EDC", "RT"]].dropna()
    return df


def build_markup():
    ptc = load_oca_ptc(OCA_PTC_DIR)
    da = load_da(DA_FILE)
    rt = load_rt(RT_FILE)

    merged = ptc.merge(da, on=["Year", "Month", "EDC"], how="inner")
    merged = merged.merge(rt, on=["Year", "Month", "EDC"], how="inner")

    merged["markup_da"] = merged["PTC"] - merged["DA"]
    merged["markup_rt"] = merged["PTC"] - merged["RT"]

    merged["markup_rate_da"] = merged["markup_da"] / merged["DA"].replace(0, pd.NA)
    merged["markup_rate_rt"] = merged["markup_rt"] / merged["RT"].replace(0, pd.NA)

    merged = merged.sort_values(["Year", "Month", "EDC"]).reset_index(drop=True)

    cols_to_round = [
        "PTC", "DA", "RT",
        "markup_da", "markup_rt",
        "markup_rate_da", "markup_rate_rt"
    ]
    merged[cols_to_round] = merged[cols_to_round].round(2)

    merged.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(merged)}")
    print(merged.head(20))

    print("\nCheck whether DA and RT are identical:")
    same_count = (merged["DA"] == merged["RT"]).sum()
    print(f"DA == RT rows: {same_count} / {len(merged)}")


if __name__ == "__main__":
    build_markup()