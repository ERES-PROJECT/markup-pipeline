import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "regression_data.csv"
csv_output = DATA_DIR / "markup_by_edc_regression_sample.csv"
html_output = DATA_DIR / "markup_by_edc_regression_sample.html"

# Load regression sample
df = pd.read_csv(input_file)

# Keep valid observations
df = df.dropna(subset=["EDC", "real_markup"])

results = []

for edc, group in df.groupby("EDC"):
    total = len(group)

    positive_count = (group["real_markup"] > 0).sum()
    negative_count = (group["real_markup"] < 0).sum()
    zero_count = (group["real_markup"] == 0).sum()

    results.append({
        "EDC": edc,
        "Mean Markup": group["real_markup"].mean(),
        "Median Markup": group["real_markup"].median(),
        "Min Markup": group["real_markup"].min(),
        "Max Markup": group["real_markup"].max(),
        "Std. Dev.": group["real_markup"].std(),
        "N": total,
        "Above PTC": positive_count,
        "Below PTC": negative_count,
        "Equal PTC": zero_count,
        "Share Above PTC": positive_count / total,
        "Share Below PTC": negative_count / total
    })

summary = pd.DataFrame(results)

# Round numerical values
for col in [
    "Mean Markup",
    "Median Markup",
    "Min Markup",
    "Max Markup",
    "Std. Dev."
]:
    summary[col] = summary[col].round(4)

# Save original numerical data as CSV
# Round shares to 4 decimal places for CSV
summary["Share Above PTC"] = summary["Share Above PTC"].round(4)
summary["Share Below PTC"] = summary["Share Below PTC"].round(4)

summary.to_csv(csv_output, index=False)

# Create display version
display = summary.copy()

display["Share Above PTC"] = (
    display["Share Above PTC"] * 100
).round(2).astype(str) + "%"

display["Share Below PTC"] = (
    display["Share Below PTC"] * 100
).round(2).astype(str) + "%"

# Create HTML table
html = """
<html>
<head>
<title>Markup Summary by EDC</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 40px;
}

h2 {
    margin-bottom: 20px;
}

table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    border: 1px solid #cccccc;
    padding: 8px 10px;
    text-align: center;
}

th {
    background-color: #f2f2f2;
}

tr:nth-child(even) {
    background-color: #fafafa;
}
</style>
</head>

<body>

<h2>Real Markup Summary by EDC</h2>

""" + display.to_html(index=False, border=0) + """

</body>
</html>
"""

with open(html_output, "w") as f:
    f.write(html)

print(display.to_string(index=False))
print()
print("CSV saved to:", csv_output)
print("HTML table saved to:", html_output)