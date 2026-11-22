# File formats

All arrays of one evaluation set have the same row order: the order of the released file of that
set. `rows/<set>.npz` gives the identity of every row, so scores can be joined to the released
data without this repository.

## Row files: `rows/<set>.npz`

| Key | Shape | Content |
| --- | --- | --- |
| `ids` | rows | position of the row in the released file (for Spica and Nebula `<split>:<row number>`) |
| `image` | rows | index of the image in the catalog of distinct images |
| `candidate` | rows | index of the candidate caption in the catalog of distinct texts |
| `references` | rows × 10 | indices of the reference captions, padded |
| `counts` | rows | number of references |
| `target` | rows | human score; absent in the repository, added by `setup_data.py` to `feature_cache_v2/<set>.npz` |

`rows/picture_ids.npy` maps the image index to an image identity (equal for two files with equal
pixels) and `rows/pictures.json` gives the SHA-1 of the decoded pixels for every identity. The
catalog itself (the texts and the image paths) is not shipped. `rows/index.json` gives, per set,
the number of rows, the official source of its human scores and the SHA-256 of the score vector.

`rows/<set>_pairs.json` (PASCAL-50S and FOIL) lists the pairs of rows that form one two-way
choice, and which row is preferred.

## Per-sample scores: `runs/probe/models/<model>/<training set>/`

| File | Content |
| --- | --- |
| `predictions/<set>.npy` | float32, rows × columns: the score of every row of `<set>` |
| `report.json` | `labels`, the name of every column; for the ridge models also the selected regularization and the validation score of every fit |
| `weights.pt` | ridge models only: the fitted coefficients (in the archive of further scores) |

`<model>` is

| Directory | Models (columns) |
| --- | --- |
| `external/classic` | BLEU-1, BLEU-4, ROUGE-L, CIDEr, METEOR, SPICE |
| `external/pacs`, `pacs_l14`, `pacspp`, `pacspp_l14` | PAC-S and PAC-S++ with and without references |
| `baselines/none` | CLIP-S, RefCLIP-S and text-only baselines |
| `external/deneb_s<seed>` | DENEB trained on the released, deduplicated and random training set |
| `external/pearl_s<seed>` | Pearl trained on the released, deduplicated, random and matched training set |
| `external/rigel` | the released Rigel checkpoint |
| `external/vlm*` | the zero-shot judges, without and with references |
| `siglip/<training set>`, `clip/<training set>` | ridge regression on frozen features: the released training set (`released`), the five cross-fitting folds (`heldout0` to `heldout4`), and for Spica fits without the samples of one evaluation set (`without_<set>`) or of all (`clean_all`) |
| `ridge_variant_<backbone>/<training set>` | ridge regression on the released, deduplicated and random training set |
| `mlp_siglip_s<seed>`, `mlp_legacy_cached_clip_s<seed>` | the three-layer network of Polos on SigLIP features and on the original CLIP features: `released` and the five folds; `official` is the released Polos checkpoint |
| `..._variants` | the same networks on the released, deduplicated and random training set |

Every seed of a retrained metric has its own directory; nothing is averaged in the stored files.

## Scores per epoch: `runs/metrics/{pearl,deneb}/<training set>_s<seed>/epoch_scores/`

One directory per saved checkpoint (`epoch=<e>-step=<s>`), holding `<set>.npz` with the key
`score`. The files `runs/metrics/logs/score_pearl_*.log` give the validation score of every epoch,
from which the selected epoch is read.

## Judges: `runs/metrics/vlm/<model>/<free|refs>/<set>.npz`

`score` is the expected rating divided by 9, `argmax` the most likely rating, and `mass` the
probability the model puts on the ten rating tokens (used in the check of the prefilled think
block, `analysis/metrics/compare_nothink.py`). The bundle holds the two InternVL runs this check
reads; the scores of all judges are in `runs/probe/models/external/vlm*`.

## Result files: `runs/metrics/*.json`, `runs/summary/*.json`, `analysis/audit/*.json`

Plain JSON written by the scripts of [pipeline.md](pipeline.md), stage 6 and stage 1.
`paper/make_tables.py` reads nothing else.

## Filter manifests

See [../dedup/README.md](../dedup/README.md).
