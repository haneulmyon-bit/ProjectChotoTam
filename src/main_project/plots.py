"""Построение и сохранение графиков в reports/figures/."""
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg


def _save(fig, filename, dpi=120):
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / filename, dpi=dpi, bbox_inches="tight")


def plot_boxplots(df, cols=None, filename="boxplots.png"):
    """Боксплоты числовых признаков (анализ выбросов)."""
    cols = cols or cfg.OUTLIER_COLS
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, c in zip(axes, cols):
        sns.boxplot(y=df[c], ax=ax)
        ax.set_title(c)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_corr_matrix(X, filename="corr_matrix.png"):
    """Корреляционная матрица признаков."""
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, annot=True, ax=ax)
    ax.set_title("Корреляционная матрица признаков")
    fig.tight_layout()
    _save(fig, filename)
    return fig
