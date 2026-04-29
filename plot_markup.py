import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "markup_full.csv")

df = pd.read_csv(FILE)

plt.figure(figsize=(8,5))

sns.kdeplot(df["markup"], linewidth=2)

#histogram
plt.hist(df["markup"], bins=50, density=True, alpha=0.3)

plt.xlabel("Markup (price - PTC)")
plt.ylabel("Density")
plt.title("Distribution of Retail Electricity Markups")

plt.tight_layout()
plt.savefig("markup_distribution.png", dpi=300)