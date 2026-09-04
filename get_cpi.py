import requests
import pandas as pd

SERIES_ID = "CUUR0000SA0"

markup = pd.read_csv("markup_full.csv")

start_year = int(markup["Year"].min())
end_year = int(markup["Year"].max())

url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

headers = {
    "Content-type": "application/json"
}

data = {
    "seriesid": [SERIES_ID],
    "startyear": str(start_year),
    "endyear": str(end_year)
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

rows = []

for item in result["Results"]["series"][0]["data"]:
    period = item["period"]
    value = item["value"]

    if period.startswith("M") and period != "M13" and value != "-":
        rows.append({
            "Year": int(item["year"]),
            "Month": int(period[1:]),
            "CPI": float(value)
        })

cpi = pd.DataFrame(rows)

cpi = cpi.sort_values(
    ["Year", "Month"]
).reset_index(drop=True)

cpi.to_csv(
    "cpi_monthly.csv",
    index=False
)

print(cpi.head())
print(cpi.tail())

print(
    f"Saved CPI data from {start_year} to {end_year}"
)