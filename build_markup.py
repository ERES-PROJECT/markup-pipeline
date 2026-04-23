import pandas as pd
import glob
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RETAIL_FILE = os.path.join(BASE_DIR, "retail_offers_2017_08_to_2025_04.csv")
PTC_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "ptc")

OUTPUT_MARKUP = os.path.join(BASE_DIR, "clean_markup.csv")
OUTPUT_MOMENTS = os.path.join(BASE_DIR, "markup_moments.csv")


# Step 1: Clean retail offers
def clean_retail(df):
    df = df.copy()

    df = df[~df["supplier"].str.contains(
        r"866|WWW|HTTP|UPDATED|¢|KWH",
        case=False,
        na=False
    )]

    df = df[(df["price"] > 3) & (df["price"] < 20)]

    df["EDC"] = df["EDC"].str.upper().str.strip()

    df = df.drop_duplicates(
        subset=["Year", "Month", "EDC", "supplier", "price"]
    )

    return df


# Step 2: Load PTC
def load_ptc(ptc_dir):
    files = glob.glob(os.path.join(ptc_dir, "oca_ptc_*.csv"))
    dfs = []

    for f in files:
        filename = os.path.basename(f)
        m = re.search(r"(20\d{2})-(\d{2})", filename)
        if not m:
            continue

        year = int(m.group(1))
        month = int(m.group(2))

        temp = pd.read_csv(f)

        temp = temp[["edc", "rate"]].copy()
        temp.columns = ["EDC", "PTC"]

        temp["Year"] = year
        temp["Month"] = month

        temp["EDC"] = temp["EDC"].str.upper().str.strip()

        temp["PTC"] = (
            temp["PTC"]
            .astype(str)
            .str.extract(r"([-]?\d+\.?\d*)", expand=False)
        )

        temp["PTC"] = pd.to_numeric(temp["PTC"], errors="coerce")

        temp = temp.dropna(subset=["PTC"])

        dfs.append(temp)

    return pd.concat(dfs, ignore_index=True)


# Step 3: Merge + Markup
def build_markup():
    retail = pd.read_csv(RETAIL_FILE)
    retail = clean_retail(retail)

    ptc = load_ptc(PTC_DIR)

    df = retail.merge(ptc, on=["Year", "Month", "EDC"], how="inner")

    # formula
    df["markup"] = df["price"] - df["PTC"]

    df.to_csv(OUTPUT_MARKUP, index=False)

    print("Clean markup saved.")
    print(df.head())

    return df


# Step 4: Compute moments
def compute_moments(df):

    summary = (
        df.groupby(["EDC", "Year", "Month"])
        .apply(lambda g: pd.Series({
            "n_offers": len(g),

            "mean_markup": g["markup"].mean(),
            "var_markup": g["markup"].var(),

            "share_above_ptc": (g["markup"] > 0).mean(),

            "q25": g["markup"].quantile(0.25),
            "q50": g["markup"].quantile(0.50),
            "q75": g["markup"].quantile(0.75),
        }))
        .reset_index()
    )

    summary.to_csv(OUTPUT_MOMENTS, index=False)

    print("\nMoments saved.")
    print(summary.head())


# Run
if __name__ == "__main__":
    df = build_markup()
    compute_moments(df)

print(df["PTC"].isna().sum())