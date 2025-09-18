import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    return fig
