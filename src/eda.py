# src/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/raw.csv"
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figs")

os.makedirs(FIG_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def summary_stats(df):
    desc = df.describe(include='all').T
    desc['n_missing'] = df.isnull().sum()
    desc['pct_missing'] = df.isnull().mean()
    return desc

def save_summary(desc):
    desc.to_csv(os.path.join(REPORT_DIR, "summary_stats.csv"))

def save_missing(df):
    miss = df.isnull().sum().sort_values(ascending=False)
    miss.to_csv(os.path.join(REPORT_DIR, "missing_counts.csv"), header=['n_missing'])

def plot_numeric_histograms(df, max_cols=6):
    num = df.select_dtypes(include=['int64','float64'])
    for i, col in enumerate(num.columns[:max_cols]):
        plt.figure(figsize=(6,4))
        sns.histplot(num[col].dropna(), kde=False)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        fname = os.path.join(FIG_DIR, f"hist_{col}.png")
        plt.savefig(fname)
        plt.close()

def plot_top_categoricals(df, topk=5):
    cat = df.select_dtypes(include=['object','category'])
    for i, col in enumerate(cat.columns[:topk]):
        plt.figure(figsize=(6,4))
        vc = df[col].value_counts().nlargest(20)
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Value counts: {col}")
        plt.tight_layout()
        fname = os.path.join(FIG_DIR, f"bar_{col}.png")
        plt.savefig(fname)
        plt.close()

def make_report(path=DATA_PATH):
    df = load_data(path)
    desc = summary_stats(df)
    save_summary(desc)
    save_missing(df)
    plot_numeric_histograms(df)
    plot_top_categoricals(df)
    print("EDA complete. Reports saved in 'reports/'")

if __name__ == "__main__":
    make_report()
