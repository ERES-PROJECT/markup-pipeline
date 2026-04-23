import os
import re
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PLANS_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "plans")
PTC_DIR = os.path.join(BASE_DIR, "..", "oca-pipeline", "data", "monthly", "ptc")

OUT_RETAIL = os.path.join(BASE_DIR, "retail_from_plans_clean.csv")
OUT_MARKUP = os.path.join(BASE_DIR, "markup_full.csv")
OUT_MOMENTS = os.path.join(BASE_DIR, "markup_moments.csv")
OUT_SIM = os.path.join(BASE_DIR, "simulated_moments.csv")

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


def extract_year_month(filename):
    m = re.search(r"(20\d{2})-(\d{2})", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def month_to_int(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except:
        months = {
            "JANUARY":1,"FEBRUARY":2,"MARCH":3,"APRIL":4,
            "MAY":5,"JUNE":6,"JULY":7,"AUGUST":8,
            "SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12
        }
        return months.get(str(x).upper(), None)


def map_zone(x):
    if pd.isna(x):
        return None
    x = str(x).upper()

    if "DUQUESNE" in x:
        return "DUQ"
    if "MET ED" in x or "MET-ED" in x:
        return "METED"
    if "PECO" in x:
        return "PECO"
    if "PENELEC" in x:
        return "PENELEC"
    if "WEST PENN" in x or "ALLEGHENY" in x:
        return "APS"
    if "PENN POWER" in x:
        return "APS"
    if "PPL" in x or "CITIZENS" in x or "UGI" in x:
        return "PPL"
    if "WELLSBORO" in x:
        return "PENELEC"

    return None


def load_plans():
    files = glob.glob(os.path.join(PLANS_DIR, "*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        y, m = extract_year_month(f)

        if y is None:
            continue

        df["Year"] = y
        df["Month"] = m

        df["price"] = pd.to_numeric(df["rate"], errors="coerce")
        df["EDC"] = df["edc"].apply(map_zone)

        df = df.dropna(subset=["price", "EDC"])
        df = df[df.apply(lambda r: in_range(r["Year"], r["Month"]), axis=1)]

        dfs.append(df[["Year", "Month", "EDC", "price"]])

    return pd.concat(dfs, ignore_index=True)


def load_ptc():
    files = glob.glob(os.path.join(PTC_DIR, "*.csv"))
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        y, m = extract_year_month(f)

        if y is None:
            continue

        df["Year"] = y
        df["Month"] = m

        df["PTC"] = pd.to_numeric(df["rate"], errors="coerce")
        df["EDC"] = df["edc"].apply(map_zone)

        df = df.dropna(subset=["PTC", "EDC"])
        df = df[df.apply(lambda r: in_range(r["Year"], r["Month"]), axis=1)]

        df = df.groupby(["Year","Month","EDC"], as_index=False)["PTC"].mean()

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_markup():
    retail = load_plans()
    ptc = load_ptc()

    df = retail.merge(ptc, on=["Year","Month","EDC"], how="left")

    df = df.dropna(subset=["PTC"])
    df["markup"] = df["price"] - df["PTC"]

    df.to_csv(OUT_MARKUP, index=False)
    return df


def compute_moments(df):
    def f(g):
        d = g["markup"]
        return pd.Series({
            "n_offers": len(d),
            "MeanMarkup": d.mean(),
            "Variance": d.var(),
            "ShareAbovePTC": (d > 0).mean(),
            "Q25": d.quantile(0.25),
            "Q50": d.quantile(0.5),
            "Q75": d.quantile(0.75),
        })

    out = df.groupby(["Year","Month","EDC"]).apply(f).reset_index()
    out.to_csv(OUT_MOMENTS, index=False)
    return out


def simulate_model(df):
    rows = []

    for (y,m,z), g in df.groupby(["Year","Month","EDC"]):
        prices = g["price"]

        if len(prices) < 5:
            continue

        logp = np.log(prices)
        mu = logp.mean()
        sigma = logp.std()

        N = len(prices)
        sim_price = np.random.lognormal(mu, sigma, N)

        ptc = g["PTC"].iloc[0]
        sim_markup = sim_price - ptc

        rows.append({
            "Year": y,
            "Month": m,
            "EDC": z,
            "n_offers": N,
            "MeanMarkup": sim_markup.mean(),
            "Variance": sim_markup.var(),
            "ShareAbovePTC": (sim_markup > 0).mean(),
            "Q25": np.quantile(sim_markup,0.25),
            "Q50": np.quantile(sim_markup,0.5),
            "Q75": np.quantile(sim_markup,0.75),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SIM, index=False)
    return out


if __name__ == "__main__":
    df = build_markup()

    print("Markup sample:")
    print(df.head())

    moments = compute_moments(df)
    print("\nData moments:")
    print(moments.head())

    sim = simulate_model(df)
    print("\nSimulated moments:")
    print(sim.head())

    print("\nSaved:")
    print(OUT_MARKUP)
    print(OUT_MOMENTS)
    print(OUT_SIM)