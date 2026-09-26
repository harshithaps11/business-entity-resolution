# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Debug Divas  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

We use a three-stage pipeline: **scalable sparse retrieval → learned candidate pruning → gradient-boosted pair matching**, followed by a decision rule that enforces the one-owner structure of the data. Retrieval uses hashed TF-IDF over name, address and house-number keys, including compound keys that stay rare when their parts are common. A cheap LightGBM pruner then shrinks the candidate set from 50 to **~7–8 records per Source 1 entity** while keeping 99% of the retrieved true matches. The final LightGBM matcher, which uses 53 features, reaches a **macro F0.5 of 0.9667** and a **pair-level accuracy of 98.3%** on a held-out split of 60,000 entities. Everything is learned only from the provided training data, and country is treated as an open set, so France is handled by the same code path.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA over the full training set (2.21M S1 / 5.03M S2 / 5.29M S3 records):

| Finding | Value | Consequence for the design |
|---|---|---|
| S1 entities with **no** match (singletons) | 5.6% | They must be predicted as empty lists, so a precision-oriented threshold is needed |
| Matches per S1 entity | mean 3.46, max 11 | The candidate set can be small |
| S2/S3 records that match **no** S1 (distractors) | ~26% | These are deliberate hard negatives |
| S2/S3 records matched to **more than one** S1 | **0** | One-owner constraint, used in the decision rule |
| True pairs that share the country label | **100%** | Per-country blocking loses no recall |
| Missing addresses (S2/S3) | ~3.3% | These pairs have to be matched on the name alone |

**Noise patterns observed** (all found in the data, not assumed):
- **Name:** junk prefixes (`#`, `@`, `***`, `>>`, `...`, `--`, brackets); legal-form variants (`Pvt/Private`, `Ltd/Limited`, `L.L.C.`, French `SARL/EURL/SAS`); `X dba Y` trade names; domain forms (`kempgloba.com`, `#bellmodern`); OCR digit–letter swaps (`Chemica1s`, `5ervices`, `Leapfr0g`); accents (`Állied`); token transpositions (`Private Isha Climate [Limited]`); typos (`Nafriommbal`); and **names written in native Indian scripts** (Devanagari, Telugu, Bengali, Gujarati, …) in 7.3% of S2/S3 names.
- **Address:** component reordering; abbreviations (`Rd/Road`, `Blvd`, French `R/Rue`, `BD`, `Av.`); state spelled out vs. code (`North Dakota`/`ND`); **state names in native scripts** (`महाराष्ट्र`); prefixes on house numbers (`H.NO`, `Door No`, `N°`, `##`); dropped components (PIN code, street, number); city misspellings (`Willston`).
- **Hard negatives:** a distractor often shares the entity's name and street but differs in its **house number** or carries an **extra name token**. For example, `Premier Boutique Care Holdings, 5666 Tri-County Hwy` is *not* `Premier Boutique Care, 5655 Tri-county Highway`. Some true matches also have altered house numbers, so number disagreement is evidence against a match rather than a veto, and a learned model is needed to weigh it.

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse retrieval) + learned pruning + gradient-boosted classifier + constrained assignment.

**Core Innovations:**
1. **Compound blocking keys** (order-insensitive name bigrams, compact full name, address bigrams, number×street-word). These stay rare when their parts are common, and they raised retrieval recall@50 from 79.6% to 96.0% on a validation sample.
2. **A learned candidate pruner**, trained only on retrieval-score and rank features. It removes 85% of retrieved pairs at almost no cost, which directly serves the "smaller candidate set" criterion.
3. **Competition ("reverse-rank") features and a one-owner decision rule**, which exploit the fact that each S2/S3 record belongs to at most one S1 entity.
4. **Transliteration maps learned from the training data** that align native-script tokens with their Latin forms (`praaivett→pvt`, `mhaaraassttr→mh`). No external dictionary or service is used.

---

## 3. Candidate Generation (Blocking)

**Stage A — retrieval (top-50 per S1).** Each record becomes a sparse, L2-normalised TF-IDF vector over **field-prefixed hashed keys** (2²³ hash buckets, so no vocabulary is held in memory):

| Key | Example | Weight |
|---|---|---|
| name core token | `n:kumar` | 1.0 |
| name bigram (sorted) | `N:climate isha` | 1.2 |
| compact name | `K:shamblinselectcontracting` | 1.2 |
| address token | `a:jopling` | 0.55 |
| address bigram (sorted) | `A:jopling rd` | 0.6 |
| house number | `d:4219` | 0.7 |
| number × address word | `D:701|jyoti` | 0.8 |

- Keys with a document frequency above 20,000 are left out of the product. They add fan-out but no discrimination, and the compound keys keep such records retrievable.
- **Blocks = country labels found in the data.** This is an open set, and France is simply a third block. The S1 × candidateᵀ sparse product runs in parallel row chunks, and the top-50 by cosine similarity are kept.
- It scales roughly linearly with the number of records: 1.73M test S1 against 10M S2+S3 records took 23 minutes on 10 CPU cores. An all-pairs comparison would be 1.7×10¹³ pairs.
- Parameters chosen experimentally: 4-character n-grams were tested and **rejected**, because they gave +0.0 recall at +60% time and +1.8 GB memory.

**Stage B — learned pruner (final candidate set).** A LightGBM (400 trees max, with early stopping) uses 10 cheap features: retrieval score, rank within the S1's list, score relative to the S1's best, gap to the next and to the top candidate, list size, **rank of this S1 among all S1s that retrieved the same candidate**, margin to the best competing S1, number of competitors, and source. Its threshold is set on validation to keep **99% of the retrieved true pairs**.

| Setting | Candidates / S1 (validation) | Recall ceiling | End-to-end macro F0.5 |
|---|---|---|---|
| Retrieval only (top-50) | 50.0 | 96.0% | — |
| Pruner, keep 99.8% | 12.96 | 95.8% | 0.9678 |
| Pruner, keep 99.5% | 9.59 | 95.5% | 0.9674 |
| **Pruner, keep 99% (submitted)** | **7.31** | **95.1%** | **0.9667** |
| Pruner, keep 98% | 5.47 | 94.1% | 0.9648 |
| Pruner, keep 95% | 4.15 | 91.2% | 0.9569 |

The 99% setting gives almost all of the F0.5 (−0.001 against 13 candidates) with a 6.8× smaller set than retrieval.

- **Candidate pairs generated (test, the set in `candidate_pairs.tsv`):** **14,717,929 pairs = 8.49 per S1 entity.** The reduction ratio against all within-country pairs is **≈ 99.9998%**. Retrieval produced 86.6M pairs, which the pruner cut by 83%.
- **How true matches were not lost:** (i) compound keys keep records made of common words retrievable; (ii) the compact-name key catches domain and concatenated names; (iii) the learned transliteration maps put native-script records in the same token space; (iv) the pruner threshold is set by a **recall target** on held-out data, not by a fixed K. The remaining misses are mostly records with no address and a generic name (e.g. `Great Network District`, empty address), which have no discriminative key.

---

## 4. Matching Model

**Preprocessing (all rule-based or learned from training data):** transliteration of non-Latin scripts followed by the learned token maps; stripping of junk symbols; OCR repair inside mixed alphanumeric tokens (`0→o, 1→l, 5→s, …`, keeping ordinals such as `4th`); DBA split (the trade name becomes the primary name and the owner an alias); domain unwrapping; legal-form canonicalisation into a separate field; address abbreviation canonicalisation (English and French); US and Indian state canonicalisation; extraction of house and unit numbers as their own token set.

**Features used (53):**
- **Name (23):** Levenshtein ratio, token-set / token-sort / partial ratio, Jaro-Winkler, compact-string ratio and partial ratio (for domain forms), Jaccard, **IDF-weighted overlap, IDF mass of tokens present only in the candidate or only in S1, and the largest such IDF** (catches added words such as "Holdings", "Spa"), token counts, first-token equality, alias (DBA) similarity, legal-form agreement or conflict, and script / domain / DBA flags.
- **Address (17):** Jaccard, IDF-weighted overlap and extra/missing IDF mass, token-set and partial ratio, empty-address flag, **house-number features** (Jaccard, number of shared numbers, S1's first number present in the candidate, extra numbers in the candidate, **near-miss** such as 5655 vs 5665, conflict flag), state agreement and missing flag.
- **Other (13):** retrieval cosine score, rank, relative score, list size, **reverse rank and margin against competing S1 entities**, number of competitors, source (S2/S3), and the pruner probability.

No feature encodes the country, so the model applies unchanged to France.

**Model selection** (same features; entity-level split with 150k S1 for training and 60k disjoint S1 for validation; the macro F0.5 includes true pairs lost in blocking and all singletons):

| Model | AUC | Avg. precision | Pair accuracy | **Macro F0.5** | Fit time |
|---|---|---|---|---|---|
| Logistic regression | 0.9899 | 0.9888 | 94.96% | 0.9286 | 48 s |
| MLP (deep net, 128-64-32, ReLU, Adam) | 0.9982 | 0.9980 | 97.99% | 0.9634 | 184 s |
| **LightGBM (selected)** | **0.9988** | **0.9986** | **98.28%** | **0.9667** | 93 s |

**Model type:** LightGBM gradient-boosted trees (MIT licence), with 127 leaves, learning rate 0.05, row/column subsampling of 0.8, and early stopping on validation log-loss. It has far fewer than 8B parameters. We chose it for the best F0.5, robustness to the mixed-scale, sentinel-valued features, and fast CPU inference (14.7M test pairs scored in about 3 minutes). The MLP came close but did not beat it.

**Most important features (LightGBM gain share):** pruner probability 46.1%, extra house numbers in the candidate 8.3%, **competition margin 7.5%**, house-number Jaccard 6.4%, IDF mass of extra name tokens 3.7%, largest IDF of an extra name token 2.9%, legal-form equality 2.9%, address token-set ratio 2.4%, house-number conflict 2.0%, compact-name partial ratio 1.9%. This matches the EDA: hard negatives are separated mainly by **house numbers** and **extra name tokens**.

**Decision rule and threshold selection:**
1. Keep pairs with probability ≥ τ.
2. Apply the **one-owner constraint**: a candidate claimed by several S1 entities goes only to the one with the highest probability.
3. An S1 with no surviving pair is predicted as a singleton.

τ is chosen by **directly maximising macro F0.5 on the validation entities**, scanning 0.30–0.95 in steps of 0.025, which gives **τ = 0.725**. The high threshold reflects the precision weighting of F0.5 and protects singletons.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, held-out validation of 60,000 S1 entities):** **0.9667**
- Pair-level: AUC 0.9988, average precision 0.9986, accuracy 98.28%.
- Test predictions: 5.50M matches, 3.18 per S1 (training ground truth has 3.46), and 6.2% predicted singletons (training has 5.6%). The statistics are consistent across countries, including France, which was not in training: matches per S1 are US 3.20 / India 3.17 / France 3.14, and singletons are 6.2% / 6.3% / 6.1%.
- Both output files pass `utils/validate_submission.py --check-ids`.

- **Common false positives (wrong merges):** hard-negative distractors that share the entity's name and street but have a different house number or one extra, rare name token (e.g. "Premier Boutique Care **Holdings**, 5666 Tri-County Hwy" vs. "Premier Boutique Care, 5655 Tri-county Highway"), and generic names in the same city when the candidate's address is missing.
- **Common false negatives (missed matches):** (i) about 4% of true pairs are never retrieved: records with an empty address and a generic or heavily misspelled name; (ii) true matches whose house number was altered by the noise process (e.g. "Bell Modern Mines, 622 9th Ave" vs. "#347 9th Ave"), which look exactly like hard negatives and are rejected by the precision-oriented threshold; (iii) native-script names whose words are not in the learned transliteration map, combined with a truncated address.

---

## 6. Conclusion

A retrieval + pruning + boosting pipeline, built around the structure of the data (compound keys, a learned pruner, house-number and extra-token evidence, and one-owner competition), resolves 1.7M test entities against 10M records on one CPU machine. It reaches a validation macro F0.5 of 0.967 with only ~8 candidates per entity. The main lessons were: (1) blocking keys must stay discriminative when individual words are common; (2) most of the difficulty lies in hard negatives that differ only by house number or an added word, so explicit numeric and IDF-difference features matter more than generic string similarity; (3) the candidate-set size can be cut sharply with a learned pruner at almost no cost in F0.5.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md` for exact commands):

```
src/
  to_parquet.py      stage 0  raw TSV -> parquet cache
  learn_translit.py  stage 1a native-script -> Latin token maps (train ground truth only)
  normalize.py       name/address normalisation rules
  preprocess.py      stage 1b parallel normalisation of all six tables
  blocking.py        hashed TF-IDF keys + per-country sparse top-K retrieval
  run_blocking.py    stage 2  retrieval for a split (checkpointed per country)
  features.py        53 pairwise + competition features
  prune.py           learned candidate pruner (defines the final candidate set)
  decide.py          threshold + one-owner assignment; macro F0.5 metric
  train.py           stage 3  entity-level split, pruner, model comparison, tuning
  predict.py         stage 4  test inference -> output/candidate_pairs.tsv, output/matching_results.tsv
run_all.sh           runs every stage end to end
requirements.txt     pinned versions
```

Entry point: `bash code/business_entity_resolution/run_all.sh <dataset_dir> <work_dir> <output_dir>`.

Runtime on 16 cores / 16 GB RAM: about 25 minutes of blocking per split, 12 minutes of training, and 6 minutes of inference.

### B. Additional Results

**Blocking design iterations (validation sample of 50k S1 against the full candidate pool, recall of true pairs within the top-K):**

| Variant | R@10 | R@30 | R@60 |
|---|---|---|---|
| Unigram keys, document-frequency cap 4,000 | 70.5% | 76.0% | 79.6% |
| + compound keys, cap 20,000 | 89.4% | 94.8% | 96.2% |
| + character 4-grams (rejected) | 89.3% | 94.8% | 96.2% |

**Compliance:** no external data, APIs, geocoders or pretrained models are used. The transliteration maps and every model are learned only from the provided training files. Libraries: numpy, pandas, scipy, scikit-learn, joblib (BSD); LightGBM, RapidFuzz (MIT); pyarrow (Apache-2.0); Unidecode (GPL-2.0, used only as a text-romanisation utility, not as a model).
