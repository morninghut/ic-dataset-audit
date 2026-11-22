# Duplicated evaluation samples in Polaris, Nebula and Spica

Code, filter lists and per-sample results for the paper

> *Do learned image captioning metrics generalize? Auditing duplicated samples in human-judgment benchmarks*, Xiaochen Zhang, Jin Li, Chunlong Xia, and Longjiang Guo (2026).

The paper audits the three recent collections of human judgments for image captioning. Samples
of their test sets are also in the training sets of the sibling collections, with the same image,
candidate caption and human ratings: the Spica training set contains 68.4% of the Polaris test
set. This repository lets you

1. **remove these samples** from a training set with one command,
2. **check the numbers of the paper** on a CPU: the tables and number macros from the result files,
   and the result files from the stored score of every metric on every sample, and
3. **rerun any stage**: the audit of the raw data, the scoring of the metrics, the trainings.

We do not redistribute the datasets. The repository holds no image, no caption and no human
score of Polaris, Nebula, Spica, Composite or Flickr8k; where scores are needed, a script reads
them from the official releases. (One file, `rows/pascal50s_pairs.json`, records for each
PASCAL-50S pair which of the two captions the annotators preferred.)

## 1. Clean a training set

```bash
python dedup/dedup.py clean spica_train --download          # writes cleaned/spica_train_dedup.csv
python dedup/dedup.py clean nebula_train --data /data/Nebula
python dedup/dedup.py clean spica_train --data /data/Spica --variant relaxed   # also near-duplicates
python dedup/dedup.py clean-any my_mixture.csv --images /data/images
python dedup/dedup.py groups polaris_test --data /data/polaris --train spica_train --out groups.csv
```

`clean` filters the released files of Spica, Nebula and Polaris by position, after checking every
row against a fingerprint. `clean-any` matches by content and works for any training file.
`groups` marks the samples of an evaluation set that a training set contains, for metrics that
cannot be retrained. Details: [dedup/README.md](dedup/README.md). `dedup.py` needs only numpy
(pyarrow for parquet, Pillow for `clean-any`).

## 2. Check the numbers of the paper

| Step | What is checked | Needs | Time |
| --- | --- | --- | --- |
| `python reproduce.py tables` | the 23 tables and 300 number macros of the manuscript, from the result files | this repository | 2 seconds |
| `python reproduce.py analysis` | the result files, from the score of every metric on every sample | `per_sample_scores.tar` (121 MB), `python setup_data.py` | about 1 hour |
| `python reproduce.py figures` | the figures that show numbers | as above | 20 seconds |

```bash
pip install -r requirements.txt            # numpy, scipy, matplotlib, pyarrow, pillow; no GPU library
python reproduce.py tables

# per-sample scores: download per_sample_scores.tar from Hugging Face into this directory
# https://huggingface.co/datasets/hutmorning/ic-dataset-audit
hf download hutmorning/ic-dataset-audit per_sample_scores.tar --repo-type dataset --local-dir .
tar -xf per_sample_scores.tar
python reproduce.py files                  # optional: SHA-256 of every unpacked file
python setup_data.py                       # reads the human scores from the official releases
python reproduce.py analysis
```

`reproduce.py analysis` reruns the bootstrap over images, the paired tests, the differences
between groups of samples and the per-epoch curves, and compares each result file with the one
shipped. The seeds are fixed, so the numbers agree to the last digit; the script prints the
largest difference per file. `reproduce.py tables` then regenerates `paper/generated/` and
compares it byte by byte with `paper/expected/`, the files from which the manuscript was compiled.
The counts of the audit (shared samples and images, structure of the collections) come from the
datasets themselves, stage 1 below, and are not recomputed here.
The nine steps of `analysis` run side by side, and `--only <step>` runs a single one. Each is one
process, except `pairwise`, which uses `--workers` processes. On our 96-core machine the run took
55 minutes, the time of the longest step, `groups`; `pairwise` took 20 minutes with 44 workers and
the other steps between 1 and 25 minutes.

`setup_data.py` reads only score columns (about 190 MB of traffic and 2 to 3 minutes; the parquet files of
Nebula and Composite are read with range requests, so no image is downloaded) and checks every
vector of scores against a SHA-256 digest. With local copies of the datasets it needs no network:
`python setup_data.py --polaris DIR --spica DIR --nebula DIR --composite FILE --flickr8k DIR`.

### What the archives hold

| Path | Content |
| --- | --- |
| `runs/probe/models/<model>/<training set>/predictions/<set>.npy` | score of every metric, judge and retrained model on every sample of every evaluation set; `report.json` next to it names the columns |
| `runs/metrics/pearl/<training set>_s<seed>/epoch_scores/` | scores of every saved epoch of the twelve Pearl trainings |
| `runs/metrics/deneb/<training set>_s<seed>/epoch_scores/` | the same for the nine DENEB trainings |
| `runs/metrics/vlm/` | the InternVL judge with and without a prefilled think block: score, most likely rating, probability on the rating tokens |

`per_sample_scores.tar` holds what `reproduce.py analysis` reads. `further_scores.tar`
(411 MB) holds the remaining stored scores, which no number of the manuscript needs: FOIL,
PASCAL-50S and the validation sets for every model, further seeds, and the fitted ridge weights.
Both archives, `SHA256SUMS`, and the released checkpoints are hosted on Hugging Face at
[`hutmorning/ic-dataset-audit`](https://huggingface.co/datasets/hutmorning/ic-dataset-audit).

The rows of a score file are the rows of `rows/<set>.npz`, which are the rows of the released file
in released order (`ids`). `rows/pictures.json` gives the pixel hash of every image, and
`dedup/manifests/evaluation_<set>.npz` the group of every sample relative to each training set.
Formats: [docs/formats.md](docs/formats.md).

## 3. Rerun a stage

Everything below needs the datasets, and from stage 2 on a GPU. The scripts are the ones we ran;
[docs/pipeline.md](docs/pipeline.md) gives the order and the directories they expect,
[docs/scripts.md](docs/scripts.md) lists every script, and [docs/environments/](docs/environments/)
the packages of each Python environment. Paths are relative to the repository root, or to
`CAPAUDIT_ROOT` if set.

| Stage | Scripts | Needs |
| --- | --- | --- |
| 1. Audit: pixel hashes, shared samples, structure of the collections | `analysis/audit/`, `analysis/summary/structure.py`, `analysis/metrics/near_duplicates.py` | CPU |
| 2. Row files and frozen features | `Polos-revised/revision/cache_features.py`, `analysis/probe/build_cache_v2.py` | GPU |
| 3. Ridge and MLP models on frozen features, cross-fitting | `analysis/probe/` | GPU |
| 4. Training: Pearl on four training sets and DENEB on three, three seeds each | `analysis/metrics/run_pearl.sh`, `run_deneb.sh` | GPU; one Pearl training took us 5 to 7 hours on an RTX 4090 |
| 5. Scoring: text-overlap metrics, CLIP-S, PAC-S, Polos, DENEB, Pearl, Rigel, zero-shot judges | `analysis/metrics/score_*.py` | GPU |
| 6. Statistics | `reproduce.py analysis` | CPU |
| 7. Tables, numbers, figures | `reproduce.py tables`, `figures` | CPU |

Polos, Pearl, DENEB and Rigel are used at the commits in `external_patches/`: DENEB and Rigel
unchanged; Pearl with a patch that skips decoding the training images, which its model does not
read, and loads the feature files once per process; Polos with the additions of `Polos.patch`
(evaluation on all sets, a SigLIP encoder) and the package `Polos-revised/revision/`.

## Layout

```
dedup/                 dedup.py and the filter manifests (stand-alone)
reproduce.py           checks of section 2
setup_data.py          human scores from the official releases into the row files
rows/                  one row file per set, without scores; pixel hashes of the images
runs/                  result files (repository) and per-sample scores (bundle)
analysis/audit/        which samples and images the collections share
analysis/probe/        ridge and MLP models on frozen features; shared statistics code
analysis/metrics/      preparation, training and scoring of the public metrics; ranking protocol
analysis/summary/      groups of samples, components of Kendall's tau, case studies
analysis/thirdparty/   check of the training files released with EXPERT
paper/                 make_tables.py, the figure scripts and the expected output
external_patches/      commits of the public repositories we ran, and our patches
docs/                  pipeline, script index, file formats, package lists of the environments
```

## Datasets and their revisions

| Dataset | Source | Revision |
| --- | --- | --- |
| Polaris | `yuwd/Polaris` (Hugging Face) | `e527aa8` |
| Nebula | `Ka2ukiMatsuda/Nebula` | `a4c99b1` |
| Spica | `hiranohachiman/Spica` | `c382725` |
| Composite | `yuwd/Composite` | `d6c63cb` |
| Flickr8k-Expert, Flickr8k-CF | the files prepared by the PAC-S authors, which the Polos repository links to; `setup_data.py` reads the copy in `hjkim811/EXPERT` (GitHub) | `7fbfb00` |

The manifests in `dedup/manifests/` and the digests in `rows/index.json` are tied to these
revisions. If a dataset is revised, `dedup.py clean` refuses the new file; `clean-any` still works.

## License

The code in this repository is released under the [MIT License](LICENSE). The filter manifests
(`dedup/manifests/`), row indices (`rows/`), and result files (`runs/`) are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`Polos-revised/revision/` builds on Polos (Clear BSD License). The fonts in `paper/fonts/` are Inter
(SIL Open Font License).

## Citation

```bibtex
@article{zhang2026icdatasetaudit,
  title   = {Do learned image captioning metrics generalize? Auditing duplicated samples in human-judgment benchmarks},
  author  = {Zhang, Xiaochen and Li, Jin and Xia, Chunlong and Guo, Longjiang},
  year    = {2026},
  note    = {Submitted to Pattern Recognition},
  url     = {https://github.com/morninghut/ic-dataset-audit}
}
```
