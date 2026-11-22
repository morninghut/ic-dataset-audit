# Pipeline

How the result files were produced, stage by stage. Stages 6 and 7 run from the per-sample
bundle on a CPU and are what `reproduce.py` executes. Stages 1 to 5 need the datasets, and from
stage 2 on a GPU; they are documented so that any of them can be rerun, and each script states
its usage and its output files in its docstring ([scripts.md](scripts.md)).

The Python scripts are the ones that produced the results, with the absolute paths of our
machines replaced by the repository root. The shell scripts (`queue*.sh`, `pipeline.sh`,
`finalize*.sh`) are the unattended queues of our two machines; read them as the exact record of
commands and order, not as a launcher for another machine. They call one script that is not
included, `pairwise_tables.py`, which printed progress reports from the ranking file.

```diagram
 datasets ──▶ 1 audit ──▶ 2 features ──▶ 3 ridge / MLP ─────────────┐
                 │             │                                    │
                 │             └──▶ 4 training (Pearl, DENEB) ──┐   │
                 │                                              ▼   ▼
                 └──▶ dedup manifests        5 scoring ──▶ per-sample scores (bundle)
                                                                │
                                                                ▼
                                    6 statistics ──▶ result files (runs/**/*.json)
                                                                │
                                                                ▼
                                    7 tables, numbers and figures of the manuscript
```

## Names used in the code

| In the code | In the paper |
| --- | --- |
| `pool` | training set |
| `picture` | image, identified by the SHA-1 of its decoded RGB pixels |
| `released` | the training set as released |
| `pairclean` | the deduplicated training set |
| `randomdrop` | the random control: as many rows removed uniformly |
| `matcheddrop` | the matched control: as many rows removed, of the kind deduplication removes |
| `duplicated` / `image_only` / `unseen` | groups of evaluation samples: the training set holds the same image and candidate / the image only / neither |
| `floor`, `above` | samples at the lowest score of an evaluation set, and the others |
| `feature_cache_v2/` | row files and frozen features ("step-2 cache" in some docstrings) |
| `runs/probe/models/<model>/<pool>/predictions/<set>.npy` | per-sample scores of a model trained on `<pool>`; `external/` holds the public metrics |

## Directories the scripts expect

All below the repository root, or below `CAPAUDIT_ROOT` if it is set.

| Path | Content | Created by |
| --- | --- | --- |
| `datasets_spica/`, `datasets_nebula/` | the Hugging Face releases of Spica and Nebula | download |
| `Polos-revised/` | Polos at the commit of `external_patches/Polos.txt`, with `Polos.patch` applied and the `revision/` package of this repository; `data_en/` inside it holds Polaris and the evaluation sets as prepared by the Polos repository. `CAPAUDIT_POLOS_HOME` overrides the location | clone |
| `external/Pearl`, `external/DENEB`, `external/Rigel` | the public repositories at the commits of `external_patches/` | clone |
| `external/envs/<name>/` | one Python environment per code base; package lists in [environments/](environments/) | `pip install -r` |
| `feature_cache_v2/` | `catalog.json` (texts and image paths), `<set>.npz` row files, frozen features | stage 2 |
| `pearl_data/` | one image directory and one CSV per set, in the format of Pearl | stage 4 |
| `features/` | features of the four encoders of Pearl, and of PAC-S | stages 4, 5 |
| `runs/` | checkpoints, per-sample scores, result files | stages 3 to 6 |

## 1. Audit of the released data

Directory `analysis/audit/`, CPU only.

1. `image_inventory.py <group>` decodes every image file and writes its pixel hash
   (`inventory_<group>.json`).
2. `overlap.py` matches images across collections; `eval_leakage.py` counts, for every evaluation
   set and training set, the samples with the same image, the same image and candidate, and the
   same human score; `csv_audit.py` and `row_types.py` describe the Spica files;
   `release_metadata.py` records licenses and revisions.
3. `analysis/summary/structure.py` counts images, image-caption pairs and ratings per collection,
   and `analysis/metrics/near_duplicates.py` estimates what exact matching misses.

Outputs shipped: `analysis/audit/*.json`, `runs/summary/structure.json`,
`runs/metrics/near_duplicates.json`.

## 2. Row files and frozen features

1. `python -m revision.cache_features` (in `Polos-revised/`) builds the catalog of distinct texts
   and images of Polaris and the classical evaluation sets and extracts SimCSE, CLIP and SigLIP
   features.
2. `analysis/probe/build_cache_v2.py` adds Spica and Nebula; the same extractor is then run on
   `feature_cache_v2`.

A row file `<set>.npz` holds, per row of the released file, the index of the image, of the
candidate and of the references in the catalog, the group of images used for the bootstrap, and
the human score. This repository ships the row files without the score (`rows/`);
`setup_data.py` restores it.

## 3. Ridge regression and the MLP on frozen features

Directory `analysis/probe/`.

1. `gram_pass.py <backbone> <pool> <partition>` accumulates the sufficient statistics of a
   training set in additive pieces, so that every subset below is a sum of pieces.
2. `run_models.py` fits the ridge models: the released training set and five cross-fitting folds
   (each evaluation sample is scored by a model that did not train on it).
   `make_variants.py` and `run_variants.py` fit the deduplicated and the random training sets.
3. `make_subsets.py`, `train_folds.sh`, `train_variants.sh` and `predict_mlp.py` do the same for
   the three-layer network of Polos (`python -m revision.train`).
4. `analyze_effects.py` holds the statistics shared by later stages: groups of samples, bootstrap
   over images, differences between models.
5. `export_filters.py` writes the lists of duplicated samples, from which
   `dedup/manifests/` is built.

## 4. Training the public metrics

Directory `analysis/metrics/`. Pearl and DENEB are trained with their released code and
hyper-parameters; only the training file changes.

1. `prep_pearl_data.py`, `prep_deneb_data.py`, `prep_matcheddrop.py` write the training files
   (released, deduplicated, random, matched); `prep_pearl_run.py` writes the configurations.
   The files are the ones `dedup/dedup.py clean` writes (checked row by row).
2. `extract_features.py <clippp|blip2|beit3|stella>` runs the four frozen encoders of Pearl once
   per distinct image and text.
3. `run_pearl.sh <released|pairclean|randomdrop|matcheddrop> <seed>` and
   `run_deneb.sh <released|pairclean|randomdrop> <seed>`; seeds 12, 13 and 14.

## 5. Scoring

Every script writes one `.npz` per evaluation set, with rows in the order of the row files.

| Metric | Script |
| --- | --- |
| BLEU, ROUGE-L, CIDEr, METEOR, SPICE | `score_classic.py` |
| CLIP-S, PAC-S, PAC-S++ and their reference versions | `score_pacs.py`, `analysis/probe/baselines.py` |
| Polos (released checkpoint and retrained) | `analysis/probe/predict_mlp.py`; the column `official` is the released checkpoint |
| DENEB, Pearl (selected checkpoint of every training) | `score_deneb.py`, `score_pearl.py`, with `best_checkpoint.py` |
| DENEB, Pearl (every saved epoch) | `analysis/summary/deneb_epoch_sweep.sh`, `epoch_sweep.py` |
| Rigel (released checkpoint) | `score_rigel.py` |
| Zero-shot judges | `score_vlm.py`, model chosen with `VLM_MODEL` |

`register.py` then copies the scores to `runs/probe/models/external/<name>/predictions/`, where
the statistics read them. `check_classic.py`, `check_pacs.py`, `check_trained.py` and
`check_rigel.py` compare every metric with the numbers of its paper under the published protocol.

## 6. Statistics (`python reproduce.py analysis`)

| Step | Script | Result file |
| --- | --- | --- |
| `leakage_pearl`, `leakage_deneb` | `analysis/metrics/leakage_seeds.py` | `runs/metrics/leakage_{pearl,deneb}.json` |
| `epoch_sweep`, `deneb_epochs` | `analysis/metrics/epoch_sweep_report.py`, `analysis/summary/deneb_epoch_report.py` | `runs/metrics/epoch_sweep.json`, `runs/summary/deneb_epoch_sweep.json` |
| `rigel_groups` | `analysis/metrics/rigel_groups.py` | `runs/metrics/rigel_groups.json` |
| `nothink` | `analysis/metrics/compare_nothink.py` | `runs/metrics/nothink_check.json` |
| `pairwise` | `analysis/metrics/pairwise.py` | `runs/metrics/pairwise_s12-13-14.json` |
| `components` | `analysis/summary/components.py` | `runs/summary/components.json` |
| `groups` | `analysis/summary/groups.py` | `runs/summary/groups.json` |

All intervals are percentile bootstrap intervals over images (the images of an evaluation set
are resampled, with all their samples); the seeds are fixed in the scripts.

The case studies (`analysis/summary/cases.py`, `dup_examples.py`) quote captions and are
therefore not rerun by `reproduce.py`; their outputs `runs/summary/cases.json` and
`dup_examples.json` hold identifiers and numbers only.

## 7. Tables, numbers and figures (`python reproduce.py tables`, `figures`)

`paper/make_tables.py` reads the result files and writes `paper/generated/`: one `.tex` file per
table and `numbers.tex`, which defines the numbers quoted in the text as macros. The scripts in `paper/figures/` draw the figures from the same
result files; `fig_examples.py` and `fig_cases.py` also need the images.
