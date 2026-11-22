# Scripts

Generated from the docstrings. Each script documents its usage and its output files at its top.

## `analysis/audit/`

Data audit: which samples and images the collections share.

| Script | Purpose |
| --- | --- |
| `benchmark_leakage.py` | Per-benchmark leakage masks: which evaluation rows use an image that is physically present in a training pool (Spica train, Polaris train, Nebula train). |
| `clip_probe.py` | How much of a benchmark's correlation comes from trivially separable rows? |
| `common.py` | Shared paths and helpers for the Spica / Nebula data audit. |
| `composite_control.py` | Control for image domain when reading the Composite leakage gap. |
| `csv_audit.py` | Audit the released Spica CSVs: size, composition, split leakage and label structure. |
| `eval_leakage.py` | Sample-level leakage of every benchmark the paper reports into each training pool. |
| `image_inventory.py` | Inventory every image file: file md5, decoded-pixel hash, format and size. |
| `label_leakage.py` | Does the Spica training set contain the evaluation samples themselves? |
| `leakage_sensitivity.py` | Split cached predictions of the Polaris-trained models into rows whose image is in the Polaris training pool and rows whose image is not. |
| `nebula_probe.py` | The same trivial-row probe on the Nebula test set, the benchmark Pearl reports. |
| `overlap.py` | Match images across datasets by exact decoded-pixel hash and report overlap. |
| `provenance.py` | Where does every Spica row come from: Polaris, Nebula, or neither? |
| `release_metadata.py` | Record how each dataset and code repository is licensed and pinned. |
| `row_types.py` | Row types inside Spica: pseudo-labelled rows without references, and "wrong-image" rows whose candidate caption belongs to a different picture. |

## `analysis/probe/`

Ridge and MLP models on frozen features, cross-fitting, shared statistics code.

| Script | Purpose |
| --- | --- |
| `analyze_effects.py` | The causal leakage effect by cross-fitting, plus the decontaminated-training comparison. |
| `baselines.py` | Training-free reference points scored on every evaluation set, in the model report format. |
| `build_cache_v2.py` | Extend feature_cache_v1 with the Spica and Nebula rows (catalog + row files only). |
| `discrimination.py` | Can a metric order two captions of the same picture as well as a human annotator can? |
| `effects.py` | Metrics on arbitrary row subsets, and picture-clustered bootstrap of differences. |
| `evaluation.py` | Metrics of one or more models on the benchmark row files, with the paper's protocols. |
| `export_filters.py` | Publishable identity lists: which (picture, candidate) pairs of a training pool are duplicated in which evaluation set, and which rows of each evaluation set are duplicated in which pool. |
| `gram_pass.py` | Accumulate ridge sufficient statistics of a training pool as reusable additive pieces. |
| `lib.py` | Closed-form ridge on the frozen Polos feature groups, with additive sufficient statistics. |
| `make_subsets.py` | Row files and standardization statistics for the fold models of the MLP head. |
| `make_variants.py` | Training subsets that remove duplicated (picture, candidate) pairs, and a size control. |
| `memorization.py` | How closely does a model reproduce the human label of a pair it was trained on? |
| `models.py` | Build ridge models from the saved statistics pieces and choose the regularization. |
| `predict_mlp.py` | Score the trained MLP fold models on every evaluation set, in the ridge report format. |
| `published.py` | What the duplicated pairs are worth to the released checkpoint itself. |
| `run_models.py` | Fit every ridge model of the design and score it on all evaluation sets. |
| `run_variants.py` | Ridge fits of the pair-decontamination experiment: released vs pairclean vs randomdrop. |
| `show_effects.py` | Readable tables of effects.json. Usage: show_effects.py <pool> [classes...] |
| `summary.py` | Collect every effects.json into the tables the report uses. |
| `test_polaris.py` | Check lib.py against the released Polaris ridge run: same lambda grid, same validation tau_c. |
| `validity.py` | What the benchmarks measure once their trivial rows are removed. |

Shell scripts, as they were run on our two machines (queues, launchers): `legacy_followup.sh`, `legacy_seeds_followup.sh`, `seed44_followup.sh`, `train_folds.sh`, `train_folds_variant.sh`, `train_variants.sh`.

## `analysis/metrics/`

Public metrics and judges: data preparation, training, scoring, ranking.

| Script | Purpose |
| --- | --- |
| `beit3_padding_effect.py` | How much BEiT-3 text features move when the batch's padding changes. |
| `beit3_padding_quality.py` | Does the padding change BEiT-3's features in a way that matters? |
| `best_checkpoint.py` | Print the checkpoint the released training loop considers best. |
| `check_classic.py` | The classic metrics under each benchmark's published protocol, next to the published numbers. |
| `check_deneb_data.py` | Check the Nebula CSVs written for DENEB against the released parquet files. |
| `check_features.py` | Spot-check the extracted Pearl features: counts, shapes, norms, NaNs, and a zero-shot image-text cosine tau_c on spica_test per backbone. |
| `check_nebula_scores.py` | Look at the rows where the cache's Nebula score differs from the parquet. |
| `check_pacs.py` | PAC-S / RefPAC-S under each benchmark's published protocol, next to the published numbers. |
| `check_rigel.py` | The released Rigel checkpoint under each benchmark's published protocol, next to the paper. |
| `check_trained.py` | A retrained DENEB or Pearl under each benchmark's published protocol, next to the paper. |
| `compare_nothink.py` | InternVL judge: scores conditional on answering directly, against scores with an empty think block prefilled. |
| `epoch_sweep.py` | Score every saved epoch of Pearl trainings, not only the one validation selected. |
| `epoch_sweep_report.py` | Evaluation scores of every epoch of the Pearl trainings scored by epoch_sweep.py. |
| `extract_features.py` | Pearl's four frozen encoders, run once over the distinct pictures and texts. |
| `inspect_nebula_parquet.py` | Print the schema and one row of the released Nebula parquet files. |
| `leakage.py` | What the duplicated pairs do to a metric trained with its authors' code. |
| `leakage_seeds.py` | The duplicated-pair effect of a retrained metric across seeds, against rows it cannot have memorized. |
| `near_duplicates.py` | How many duplicated judgments does exact matching miss? |
| `pairwise.py` | Which metrics can each benchmark actually tell apart? |
| `prep_deneb_data.py` | Nebula data for DENEB, in the same picture identities as pearl_data. |
| `prep_matcheddrop.py` | A fourth Pearl training pool: a control matched to what pairclean removes. |
| `prep_pearl_data.py` | Pearl-format data for every split, keyed by picture identity. |
| `prep_pearl_run.py` | Pearl's data directory and the three training configurations. |
| `protocol.py` | The duplication-free, floor-aware protocol applied to every metric we can score. |
| `register.py` | Put externally scored metrics where the analysis tools look for predictions. |
| `released_cli.py` | Run the released Pearl / DENEB command line without its dead download. |
| `rigel_groups.py` | How much of Rigel's score on the Polaris-family test sets comes from rows it was trained on? |
| `score_classic.py` | BLEU, ROUGE-L, CIDEr, METEOR and SPICE for every split, with the pycocoevalcap scorers. |
| `score_deneb.py` | Score every evaluation split with a trained DENEB checkpoint. |
| `score_pacs.py` | PAC-S / RefPAC-S and PAC-S++ / RefPAC-S++ with the released checkpoints, for every split. |
| `score_pearl.py` | Score every evaluation split with a trained Pearl checkpoint. |
| `score_rigel.py` | Score every evaluation split with the released Rigel checkpoint. |
| `score_vlm.py` | A zero-shot vision-language judge of caption quality, for every evaluation split. |
| `verify_images.py` | Verify every picture the Pearl CSVs reference opens, once, in parallel. |

Shell scripts, as they were run on our two machines (queues, launchers): `check_pearl_fastpath.sh`, `finalize_ranking.sh`, `finalize_rigel.sh`, `finalize_step4.sh`, `pipeline.sh`, `queue.sh`, `queue_deneb_seed.sh`, `queue_pearl.sh`, `queue_pearl_seed.sh`, `queue_rigel.sh`, `queue_sweep_remote.sh`, `queue_vlm.sh`, `rerank.sh`, `run_deneb.sh`, `run_pearl.sh`, `sync_remote.sh`.

## `analysis/summary/`

Statistics of the paper: structure of the collections, groups, components of tau, cases.

| Script | Purpose |
| --- | --- |
| `case_data.py` | Per image-caption pair of an evaluation set: texts, individual ratings and the score of every metric as a percentile of that set, for the case studies. |
| `case_explore.py` | List candidate case studies by how metric percentiles deviate from the human percentile. |
| `cases.py` | Write the case studies shown in the paper to runs/summary/cases.json. |
| `components.py` | Split Kendall's tau over all samples into its two parts. |
| `deneb_epoch_report.py` | Kendall tau of every saved epoch of the DENEB trainings on the Polaris test set. |
| `dup_examples.py` | Concrete examples of duplicated samples for the paper. |
| `groups.py` | Sample groups by what the training set holds of them: descriptive statistics, label agreement of the duplicated pairs, and the retraining and cross-fit contrasts with the lowest-score samples removed. |
| `structure.py` | How each collection is built: images, image-caption pairs and individual ratings. |

Shell scripts, as they were run on our two machines (queues, launchers): `after_sweep.sh`, `deneb_epoch_sweep.sh`, `mlp_nebula.sh`.

## `analysis/thirdparty/`

Checks of training files released by third parties.

| Script | Purpose |
| --- | --- |
| `expert_overlap.py` | Count the test samples of Polaris and Nebula that are in the training files released with EXPERT. |

## `paper/`

Tables and numbers of the manuscript.

| Script | Purpose |
| --- | --- |
| `make_tables.py` | Write every table and every number of the paper from the result files. |

## `paper/figures/`

Figures.

| Script | Purpose |
| --- | --- |
| `common.py` | Shared look and data access for the paper's figures. |
| `fig_all_above.py` | What is left of the correlation over all samples once the lowest-score samples are set aside. |
| `fig_cases.py` | Case studies: one image-caption pair per row, with where the humans and the metrics place it. |
| `fig_components.py` | Kendall's tau over all samples is the sum of a detection term and a ranking term. |
| `fig_crossfit.py` | Cross-fit effect of having trained on a sample, by learner. |
| `fig_duplication.py` | How much of each evaluation set is already inside each training set. |
| `fig_epochs.py` | Kendall tau_c on the Polaris test set after every training epoch. |
| `fig_examples.py` | Three kinds of duplicate, each with one real example. |
| `fig_groups.py` | The gain from keeping the duplicates, within each group of Polaris test samples. |
| `fig_heatmap.py` | The re-ranking as a heat map: Kendall tau (number) and rank within the set (shade). |
| `fig_leaderboard.py` | The ranking on the Polaris test set and on Composite, as two leaderboards. |
| `fig_overview.py` | Figure 1: schematic of the study. |
| `fig_power.py` | What the evaluation sets can tell apart. |
