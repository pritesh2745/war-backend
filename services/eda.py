
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.savefig("correlation.png")
    plt.close()

def plot_attack_distribution(df):
    df[["drones", "missiles", "munitions"]].sum().plot(kind="bar")
    plt.savefig("attacks.png")
    plt.close()
