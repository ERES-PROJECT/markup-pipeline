import pandas as pd
from pathlib import Path

data_file = Path(__file__).resolve().parent.parent / "data" / "rt_pa_all_years_summary.csv"

df = pd.read_csv(data_file)

# Convert from $/MWh to cents/kWh and keep 3 decimal places
df["Average"] = (df["Average"] / 10).round(3)
df["Median"] = (df["Median"] / 10).round(3)

df.to_csv(data_file, index=False)

print("RT prices converted from $/MWh to cents/kWh.")
print(df.head())