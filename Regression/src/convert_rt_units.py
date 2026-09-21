import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGRESSION_DIR = os.path.dirname(BASE_DIR)

FILES = [
    os.path.join(REGRESSION_DIR, "rt_pa_all_years_summary.csv"),
    os.path.join(REGRESSION_DIR, "data", "rt_pa_all_years_summary.csv")
]

for file_path in FILES:
    df = pd.read_csv(file_path)

    # PJM RT price: $/MWh -> cents/kWh
    df["Average"] = (df["Average"] / 10).round(3)
    df["Median"] = (df["Median"] / 10).round(3)

    # Overwrite the original file
    df.to_csv(file_path, index=False)

    print("Converted:", file_path)

print("Done.")