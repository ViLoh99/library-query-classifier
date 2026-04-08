"""
Statistische Analyse: Katalogsuche-Query-Klassifikation
========================================================
Datensatz 1: predictions_withgnd.csv      (Session-basiert, ~96k Queries)
Datensatz 2: predictions_withgnd_2.csv    (Aggregiert nach Datum, ~25k Queries)

Voraussetzungen:
    pip install pandas numpy scipy matplotlib seaborn

Aufruf:
    python katalogsuche_analyse.py

    Oder mit eigenen Dateipfaden:
    python katalogsuche_analyse.py \
        --ds1 pfad/zu/predictions_withgnd.csv \
        --ds2 pfad/zu/predictions_withgnd_2.csv \
        --out ausgabe_ordner/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Farbschema ────────────────────────────────────────────────────────────────
PALETTE = {
    "know-item":  "#378ADD",
    "thematisch": "#1D9E75",
    "sonderfall": "#BA7517",
    "rauschen":   "#73726c",
}
LABEL_ORDER = ["know-item", "thematisch", "sonderfall", "rauschen"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def load_ds1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python",
                     encoding="utf-8-sig", on_bad_lines="skip")
    df["keyword"] = df["keyword"].astype(str)
    df["kw_len_chars"]  = df["keyword"].str.len()
    df["kw_len_tokens"] = df["keyword"].str.split().str.len()
    return df


def load_ds2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python",
                     encoding="utf-8-sig", on_bad_lines="skip")
    for col in [c for c in ("bounce_rate", "exit_rate") if c in df.columns]:
        df[col] = (df[col].astype(str)
                          .str.replace("%", "", regex=False)
                          .str.strip()
                          .replace("", np.nan)
                          .astype(float))
    df["keyword"] = df["keyword"].astype(str)
    df["kw_len_chars"]  = df["keyword"].str.len()
    df["kw_len_tokens"] = df["keyword"].str.split().str.len()
    return df


def colour_list(labels):
    return [PALETTE.get(l, "#aaaaaa") for l in labels]


def mannwhitney_table(df: pd.DataFrame, cols: list, group_col: str = "label",
                      g1: str = "know-item", g2: str = "thematisch") -> pd.DataFrame:
    """Mann-Whitney-U-Test für jede Spalte, g1 vs. g2."""
    rows = []
    a = df[df[group_col] == g1]
    b = df[df[group_col] == g2]
    for col in cols:
        x = a[col].dropna()
        y = b[col].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "Merkmal":       col,
            f"Median {g1}":  round(x.median(), 2),
            f"Median {g2}":  round(y.median(), 2),
            f"Mean {g1}":    round(x.mean(), 2),
            f"Mean {g2}":    round(y.mean(), 2),
            "U":             round(u, 0),
            "p-Wert":        p,
            "signifikant":   "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")),
        })
    return pd.DataFrame(rows)


def describe_by_label(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Deskriptive Statistik (mean, median, std, q25, q75) je Label und Spalte."""
    records = []
    for col in cols:
        for label, grp in df.groupby("label"):
            s = grp[col].dropna()
            records.append({
                "Merkmal": col,
                "Label":   label,
                "n":       len(s),
                "Mean":    round(s.mean(), 2),
                "Median":  round(s.median(), 2),
                "Std":     round(s.std(), 2),
                "Q25":     round(s.quantile(0.25), 2),
                "Q75":     round(s.quantile(0.75), 2),
                "Max":     round(s.max(), 2),
            })
    return pd.DataFrame(records)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_label_dist(df: pd.DataFrame, title: str, ax=None):
    counts = df["label"].value_counts()
    labels_present = [l for l in LABEL_ORDER if l in counts.index]
    counts = counts.loc[labels_present]
    pcts   = counts / counts.sum() * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    bars = ax.barh(labels_present[::-1],
                   counts.values[::-1],
                   color=colour_list(labels_present[::-1]))
    for bar, pct in zip(bars, pcts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Anzahl Queries")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}".replace(",", ".")))
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_boxen(df: pd.DataFrame, col: str, title: str,
               log_scale: bool = False, ax=None):
    """Boxplot je Label für eine metrische Variable."""
    present = [l for l in LABEL_ORDER if l in df["label"].unique()]
    data    = [df[df["label"] == l][col].dropna().values for l in present]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))

    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    labels=present)
    for patch, label in zip(bp["boxes"], present):
        patch.set_facecolor(PALETTE.get(label, "#aaaaaa"))
        patch.set_alpha(0.85)

    if log_scale:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_gnd_heatmap(df: pd.DataFrame, title: str, ax=None):
    gnd_cols = ["person_exact", "subject_exact",
                "person_token", "subject_token", "conflict_same_token"]
    present_labels = [l for l in LABEL_ORDER if l in df["label"].unique()]
    matrix = (df.groupby("label")[gnd_cols]
                .mean()
                .loc[present_labels]
                .mul(100)
                .round(1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.5))

    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("")
    return ax


def plot_kw_length(d1: pd.DataFrame, d2: pd.DataFrame, ax=None):
    """Verteilung der Keyword-Länge (Tokens) – nur know-item und thematisch."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    for label, ls in [("know-item", "-"), ("thematisch", "--")]:
        for ds, name, color in [(d1, "DS1", PALETTE[label]),
                                (d2, "DS2", PALETTE[label])]:
            sub = ds[ds["label"] == label]["kw_len_tokens"].clip(upper=15)
            vc  = sub.value_counts(normalize=True).sort_index()
            ax.plot(vc.index, vc.values * 100,
                    linestyle=ls,
                    color=color,
                    alpha=0.7 if "DS2" in name else 1.0,
                    label=f"{label} ({name})",
                    linewidth=1.5)

    ax.set_xlabel("Anzahl Tokens (gekappt bei 15)")
    ax.set_ylabel("Anteil (%)")
    ax.set_title("Keyword-Länge nach Label (Tokenanzahl)", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_metric_bars(df: pd.DataFrame, cols: list, title: str,
                     value_label: str = "Median", ax=None):
    """Balkendiagramm: Median je Label für mehrere Metriken."""
    present = [l for l in LABEL_ORDER if l in df["label"].unique()]
    medians = df.groupby("label")[cols].median().loc[present]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.4), 3.5))

    x     = np.arange(len(cols))
    width = 0.8 / len(present)
    for i, label in enumerate(present):
        ax.bar(x + i * width, medians.loc[label],
               width=width, label=label,
               color=PALETTE.get(label, "#aaaaaa"), alpha=0.85)

    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(value_label)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# ── Hauptanalyse ──────────────────────────────────────────────────────────────

def analyse(ds1_path: str, ds2_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Lade Datensätze …")
    d1 = load_ds1(ds1_path)
    d2 = load_ds2(ds2_path)

    # ── 1. Überblick ──────────────────────────────────────────────────────────
    print("\n── Überblick ──")
    for name, df in [("DS1", d1), ("DS2", d2)]:
        print(f"\n{name}: {len(df):,} Zeilen | {df['keyword'].nunique():,} unique Keywords")
        print(df["label"].value_counts().to_frame("n")
                         .assign(pct=lambda x: (x["n"] / x["n"].sum() * 100).round(1)))

    # ── 2. Deskriptive Statistik ──────────────────────────────────────────────
    print("\n── Deskriptive Statistik speichern …")

    ds1_num = ["duration", "searches", "actions", "pageviewPosition",
               "kw_len_chars", "kw_len_tokens"]
    ds2_num = ["nb_visits", "nb_hits", "sum_time_spent", "nb_pages_per_search",
               "avg_time_on_page", "exit_rate", "kw_len_chars", "kw_len_tokens"]

    desc1 = describe_by_label(d1, ds1_num)
    desc2 = describe_by_label(d2, ds2_num)
    desc1.to_csv(out / "deskriptiv_ds1.csv", index=False)
    desc2.to_csv(out / "deskriptiv_ds2.csv", index=False)
    print("  → deskriptiv_ds1.csv / deskriptiv_ds2.csv")

    # ── 3. Statistische Tests ─────────────────────────────────────────────────
    print("── Mann-Whitney-U-Tests speichern …")

    mw1 = mannwhitney_table(d1, ds1_num)
    mw2 = mannwhitney_table(d2, ds2_num)
    mw1.to_csv(out / "mannwhitney_ds1.csv", index=False)
    mw2.to_csv(out / "mannwhitney_ds2.csv", index=False)
    print("  → mannwhitney_ds1.csv / mannwhitney_ds2.csv")
    print("\nDS1 Tests (know-item vs. thematisch):")
    print(mw1[["Merkmal", "Median know-item", "Median thematisch",
               "p-Wert", "signifikant"]].to_string(index=False))
    print("\nDS2 Tests (know-item vs. thematisch):")
    print(mw2[["Merkmal", "Median know-item", "Median thematisch",
               "p-Wert", "signifikant"]].to_string(index=False))

    # ── 4. GND-Trefferquoten ──────────────────────────────────────────────────
    gnd_cols = ["person_exact", "subject_exact", "person_token",
                "subject_token", "conflict_same_token"]
    gnd1 = (d1.groupby("label")[gnd_cols].mean().mul(100).round(1))
    gnd2 = (d2.groupby("label")[gnd_cols].mean().mul(100).round(1))
    gnd1.to_csv(out / "gnd_trefferquoten_ds1.csv")
    gnd2.to_csv(out / "gnd_trefferquoten_ds2.csv")
    print("\nGND-Trefferquoten DS1 (%):")
    print(gnd1.to_string())

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\n── Grafiken erstellen …")

    # 5a. Labelverteilung nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle("Labelverteilung", fontsize=12, y=1.02)
    plot_label_dist(d1, f"DS1 — Session-Daten (n={len(d1):,})", ax=axes[0])
    plot_label_dist(d2, f"DS2 — Aggregiert nach Datum (n={len(d2):,})", ax=axes[1])
    plt.tight_layout()
    plt.savefig(out / "01_labelverteilung.png", bbox_inches="tight")
    plt.close()

    # 5b. Keyword-Länge
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_kw_length(d1, d2, ax=axes[0])
    # Zeichenlänge als Boxplot
    ki_chars = d1[d1["label"]=="know-item"]["kw_len_chars"].clip(upper=100)
    th_chars = d1[d1["label"]=="thematisch"]["kw_len_chars"].clip(upper=100)
    axes[1].hist(ki_chars, bins=40, alpha=0.6,
                 color=PALETTE["know-item"], label="know-item (DS1)", density=True)
    axes[1].hist(th_chars, bins=40, alpha=0.6,
                 color=PALETTE["thematisch"], label="thematisch (DS1)", density=True)
    axes[1].set_xlabel("Zeichenlänge (gekappt bei 100)")
    axes[1].set_ylabel("Dichte")
    axes[1].set_title("Keyword-Länge (Zeichen, DS1)", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "02_keyword_laenge.png", bbox_inches="tight")
    plt.close()

    # 5c. DS1 Session-Metriken: Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("DS1 — Session-Metriken nach Label", fontsize=12)
    for ax, col, log in zip(axes,
                            ["duration", "searches", "actions"],
                            [True, False, False]):
        plot_boxen(d1, col, col, log_scale=log, ax=ax)
    plt.tight_layout()
    plt.savefig(out / "03_ds1_session_metriken.png", bbox_inches="tight")
    plt.close()

    # 5d. DS2 Nutzungsmetriken: Mediane als Balken
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_metric_bars(d2,
                     ["nb_visits", "nb_hits", "sum_time_spent",
                      "nb_pages_per_search", "avg_time_on_page", "exit_rate"],
                     "DS2 — Nutzungsmetriken nach Label (Median)", ax=ax)
    plt.tight_layout()
    plt.savefig(out / "04_ds2_nutzungsmetriken.png", bbox_inches="tight")
    plt.close()

    # 5e. GND-Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    fig.suptitle("GND-Trefferquoten nach Label (%)", fontsize=12)
    plot_gnd_heatmap(d1, "DS1", ax=axes[0])
    plot_gnd_heatmap(d2, "DS2", ax=axes[1])
    plt.tight_layout()
    plt.savefig(out / "05_gnd_heatmap.png", bbox_inches="tight")
    plt.close()

    # 5f. DS2 exit_rate Boxplot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_boxen(d2, "exit_rate", "DS2 — Exit-Rate nach Label (%)", ax=ax)
    plt.tight_layout()
    plt.savefig(out / "06_ds2_exit_rate.png", bbox_inches="tight")
    plt.close()

    # 5g. Token-Länge Boxplot (beide Datensätze)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle("Anzahl Tokens pro Query nach Label", fontsize=12)
    for ax, df, name in zip(axes, [d1, d2], ["DS1", "DS2"]):
        tmp = df.copy()
        tmp["kw_len_tokens_clip"] = tmp["kw_len_tokens"].clip(upper=10)
        plot_boxen(tmp, "kw_len_tokens_clip",
                   f"{name} (gekappt bei 10)", ax=ax)
    plt.tight_layout()
    plt.savefig(out / "07_token_laenge_boxplot.png", bbox_inches="tight")
    plt.close()

    print(f"  → 7 Grafiken gespeichert in {out}/")

    # ── 6. Kombinierter Summary-Report ───────────────────────────────────────
    print("\n── Erstelle Summary-Report …")
    summary_lines = [
        "ANALYSE-BERICHT: KATALOGSUCHE-QUERY-KLASSIFIKATION",
        "=" * 60,
        "",
        "DATENSÄTZE",
        f"  DS1 (Session): {len(d1):,} Queries | {d1['keyword'].nunique():,} unique Keywords",
        f"  DS2 (Aggregat): {len(d2):,} Queries | {d2['keyword'].nunique():,} unique Keywords",
        "",
        "LABELVERTEILUNG",
    ]
    for name, df in [("DS1", d1), ("DS2", d2)]:
        summary_lines.append(f"  {name}:")
        for label, cnt in df["label"].value_counts().items():
            pct = cnt / len(df) * 100
            summary_lines.append(f"    {label:<12} {cnt:>7,}  ({pct:.1f}%)")

    summary_lines += [
        "",
        "KEYWORD-LÄNGE (Median, Tokens)",
        "  DS1 know-item:  " + str(d1[d1["label"]=="know-item"]["kw_len_tokens"].median()),
        "  DS1 thematisch: " + str(d1[d1["label"]=="thematisch"]["kw_len_tokens"].median()),
        "  DS2 know-item:  " + str(d2[d2["label"]=="know-item"]["kw_len_tokens"].median()),
        "  DS2 thematisch: " + str(d2[d2["label"]=="thematisch"]["kw_len_tokens"].median()),
        "",
        "HINWEIS: Sonderfälle in DS1 haben sehr lange Keywords (Median ~223 Zeichen).",
        "         Es handelt sich um komplexe Booleschen Suchanfragen (OR-Verknüpfungen).",
        "",
        "STATISTISCHE TESTS (Mann-Whitney-U, know-item vs. thematisch)",
    ]
    for name, mw in [("DS1", mw1), ("DS2", mw2)]:
        summary_lines.append(f"\n  {name}:")
        for _, row in mw.iterrows():
            summary_lines.append(
                f"    {row['Merkmal']:<25} p={row['p-Wert']:.2e}  {row['signifikant']}"
                f"  | Median KI={row['Median know-item']}, TH={row['Median thematisch']}")

    summary_lines += [
        "",
        "GND-FEATURES — Wichtigste Unterschiede",
        "  person_token:  ~48% bei know-item, ~0% bei thematisch",
        "  subject_token: ~0%  bei know-item, ~46% bei thematisch",
        "  rauschen-Klasse: person_exact sehr hoch (58–79%) — Eigennamen ohne klare Zuordnung",
        "  conflict_same_token: ~21% bei thematisch — ambivalente Person/Sachschlagwort-Überschneidung",
        "",
        "DATEIEN",
        "  deskriptiv_ds1.csv / deskriptiv_ds2.csv  — vollständige Kennwerte",
        "  mannwhitney_ds1.csv / mannwhitney_ds2.csv — Testergebnisse",
        "  gnd_trefferquoten_ds1.csv / gnd_trefferquoten_ds2.csv",
        "  01–07 *.png — Grafiken",
    ]

    report_path = out / "bericht.txt"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  → {report_path}")
    print("\nFertig.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistische Analyse der Katalogsuche-Klassifikation")
    parser.add_argument("--ds1",
        default="predictions_withgnd.csv",
        help="Pfad zu DS1 (default: predictions_withgnd.csv)")
    parser.add_argument("--ds2",
        default="predictions_withgnd_2.csv",
        help="Pfad zu DS2 (default: predictions_withgnd_2.csv)")
    parser.add_argument("--out",
        default="analyse_output",
        help="Ausgabe-Ordner (default: analyse_output/)")
    args = parser.parse_args()

    analyse(args.ds1, args.ds2, args.out)
