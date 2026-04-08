# library-query-classifier

Classifying library catalog search queries into **know-item** vs. **thematic** searches using GND lookup and SBERT + logistic regression.

*Developed as part of a library and information science research project, 2025.*

---

## Overview

When users search a library catalog, their intent differs fundamentally:

- **Know-item**: The user is looking for a specific work or person — e.g. an author name, a title, an ISBN
- **Thematic**: The user is exploring a topic or subject area without a specific work in mind

This project builds an automated classification pipeline for this distinction, combining rule-based classification (using the German National Authority File, GND) with a machine learning classifier (SBERT embeddings + logistic regression). The pipeline reaches a **Macro F1 of 0.832** on a held-out test set.

---

## Pipeline

```
Raw log data (CSV)
        │
        ▼
┌─────────────────────────┐
│   analysis_pipeline     │  ← GND lookup + rule-based classification
│        .ipynb           │    + SBERT training + inference
└─────────────────────────┘
        │
        ▼
output (CSV)
        │
        ▼
┌─────────────────────────┐
│ catalog_search_analysis │  ← Descriptive stats + Mann-Whitney tests
│         .py             │    + visualizations
└─────────────────────────┘
        │
        ▼
analyse_output/  (CSV tables + plots + report)
```

The notebook covers four stages:

1. **GND index** — builds a lookup dictionary from the DNB's open authority dumps (persons + subject headings), cached as `gnd_index.pkl` after the first run (~15–30 min, one-time)
2. **Rule-based classification** — high-confidence patterns (ISBN, comma notation, year tokens, GND hits) are classified deterministically
3. **ML training** — manually annotated queries are combined with high-confidence rule labels; SBERT embeddings + 8 GND features (392 dimensions total) are fed to a logistic regression classifier
4. **Inference** — new log data is classified using the saved model and GND index

The annotation tool (`annotation_tool.html`) is a standalone browser app for labeling queries with keyboard shortcuts (K / T / S / Z).

---

## Files

| File | Description |
|---|---|
| `analysis_pipeline.ipynb` | Full pipeline: GND index, rule-based + ML classification, inference |
| `annotation_tool.html` | Browser-based annotation tool (keyboard shortcuts, CSV export) |
| `catalog_search_analysis.py` | Statistical analysis: descriptive stats, Mann-Whitney tests, plots — **metrics need to be adjusted to match your data** (e.g. remove `bounce_rate`, `exit_rate` if not present in your logs) |

---

## Setup

```bash
Python 3.10+
pip install -r requirements.txt
```

**GND authority dumps** (required for index build, one-time download ~1.3 GB):

```bash
wget https://data.dnb.de/opendata/authorities-gnd-person_lds.jsonld.gz
wget https://data.dnb.de/opendata/authorities-gnd-sachbegriff_lds.jsonld.gz
```

---

## Model

| Component | Choice | Reason |
|---|---|---|
| Sentence encoder | `paraphrase-multilingual-MiniLM-L12-v2` | Good German support, small (~120 MB), fast |
| Classifier | Logistic Regression | Stable at ~1,000 samples, interpretable |
| Class weights | `balanced` | Compensates for natural know-item / thematic imbalance (~1.4:1) |

---

## Known Limitations

- **GND gaps**: Not all authors are in the GND — rare or international names are frequently missed by the lookup, leading to more queries falling through to the ML classifier
- **Short ambiguous strings**: Single tokens like `python` or `prometheus` are not reliably classifiable without additional context
- **Fuzzy lookup disabled in inference**: For performance reasons (~10h runtime on 100k queries), fuzzy matching is skipped during inference — roughly 3% of queries lose GND coverage as a result
- **Multilingual queries**: The SBERT model is multilingual, but the GND index is predominantly German; non-German author names are often not recognized as persons

---

## Outlook

- **Use confidence scores**: The classifier outputs probabilities that are currently unused — queries below a confidence threshold (e.g. < 0.65) could be flagged for manual review or targeted re-annotation
- **More training data**: ~1,000 annotated samples is sufficient but limited, especially for rare patterns (short ambiguous queries, foreign-language inputs, unusual spellings)
- **Click feedback as weak supervision**: Post-search click behavior (did the user open a result, how long did they stay?) could serve as a weak labeling signal, reducing the need for manual annotation

---

## Data & Privacy

The log data used for training and evaluation is not included in this repository (confidential). The GND authority data is published by the Deutsche Nationalbibliothek under a [CC0 license](https://www.dnb.de/EN/Professionell/Metadatendienste/opendata.html).

---

## License

MIT
