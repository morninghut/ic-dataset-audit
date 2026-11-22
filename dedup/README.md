# Removing evaluation samples from a training set

Polaris, Nebula and Spica share samples across their splits: the same image with the same
candidate caption and the same human ratings is in the training set of one collection and in a
test set of another. `dedup.py` writes training files without these samples and tells, for an
evaluation set, which samples a given training set contains.

It is one file with no dependency on the rest of the repository.

```bash
python dedup.py list                                    # what the manifests cover
python dedup.py clean spica_train --download            # fetch spica_train.csv, write cleaned/spica_train_dedup.csv
python dedup.py clean spica_train --data /data/Spica    # the same from a local copy
python dedup.py clean nebula_train --data /data/Nebula  # parquet in, parquet out; images are not decoded
python dedup.py clean polaris_train --data /data/polaris
```

Requirements: Python 3.9 or later and numpy; pyarrow for parquet files; Pillow for `clean-any`.

## What "the same sample" means

Two rows hold the same sample if their images have identical decoded pixels and their candidate
captions are identical strings. Image identifiers are not used: in Spica 13,964 images are stored
under two identifiers. The rule is strict on purpose; counting near-identical captions and
re-encoded images as well raises the four large shares by less than one percentage point.

`--variant relaxed` applies that looser rule as well. Two rows then also hold the same sample if
their images have a cosine of at least 0.97 between their SigLIP so400m-patch14-384 features (the
same photograph stored twice, or another shot of the same scene) and their captions are equal after
lower-casing and removing punctuation, or have a character similarity of at least 0.9 (difflib
ratio of the normalized strings). It removes every row that the exact rule removes and a few more;
it can also remove another shot of the same scene with a near-identical caption, which a model need
not have memorized. The experiments of the paper use the exact rule.

## Commands

| Command | Input | Output |
| --- | --- | --- |
| `clean <split>` | a released training or validation file | the file without the rows whose sample is in an evaluation set, and a report |
| `groups <split> --train <training split>` | a released evaluation file | one line per sample: `duplicated`, `image_only` or `unseen_image` |
| `clean-any <file> --images <dir>` | any CSV, JSON, JSON-lines or parquet training file | the same as `clean`, matched by content |
| `list` | | the splits and the share of duplicated samples |

`clean` and `groups` identify a row by its position in the released file. Before anything is
written, every row is compared with a fingerprint of its image identifier and candidate caption,
so a file from another revision is refused. The kept rows are written unchanged, byte for byte.

`clean-any` decodes every image once and matches by content. Use it for a mixture of several
collections, for a later revision of a dataset, or for your own training set. For the released
Spica files it writes exactly the files that `clean` writes.

### Options of `clean`

- `--variant relaxed` removes the rows that match an evaluation sample under the relaxed rule above;
  `--against` applies as for the default.
- `--variant random` and `--variant matched` write the control sets of the paper: the same number of
  rows removed at random, or removed among similar samples that are in no evaluation set
  (Spica only).
- `--against polaris_test,nebula_test` limits the evaluation sets. The default is the list used in
  the paper, which also contains the validation sets of the other collections and the classic
  benchmarks.

## Splits and revisions

| Split | Rows | Kept, exact rule | Kept, relaxed rule | Dataset revision |
| --- | --- | --- | --- | --- |
| `spica_train` | 309,386 | 266,893 | 264,015 | `hiranohachiman/Spica` @ `c382725` |
| `spica_val` | 8,773 | 6,700 | 6,568 | same |
| `nebula_train` | 26,382 | 22,271 | 21,746 | `Ka2ukiMatsuda/Nebula` @ `a4c99b1` |
| `polaris_train` | 78,630 | 65,051 | 63,287 | `yuwd/Polaris` @ `e527aa8` |

Evaluation sets covered by `groups`: `polaris_test`, `polaris_val`, `spica_test`, `spica_val`,
`nebula_test`, `nebula_valid`. For Flickr8k-Expert, Flickr8k-CF and Composite the manifests hold
the groups by sample identifier (`manifests/evaluation_<set>.npz`); no training set contains a
sample of the two Flickr8k sets, and one sample of Composite.

## Evaluating a metric that is already trained

A metric trained on a released training set cannot be retrained by everyone. Report it on the
samples whose image is not in that training set:

```bash
python dedup.py groups polaris_test --data /data/polaris --train spica_train --out groups.csv
```

`groups.csv` has one line per row of `polaris_test.csv`. Keep the rows marked `unseen_image`.

## Manifests

`manifests/` holds no image, no caption and no human score.

| File | Content |
| --- | --- |
| `training_<split>.npz` | per row: `fingerprint`, `duplicated_in` (bit mask over the evaluation sets in `index.json`), `duplicated_in_relaxed` (the same under the relaxed rule), `keep_dedup`, `keep_random`, `keep_matched` |
| `evaluation_<set>.npz` | per sample: `fingerprint` or `sample_id`, and `group_<training split>` with 0 unseen image, 1 image only, 2 duplicated |
| `evaluation_content.npz` | for every evaluation set the keys of its images and of its image-caption pairs, computed from content; used by `clean-any` |
| `index.json` | names, counts, dataset revisions and the definition of every key |

The image hash is the SHA-1 of `"<width>x<height>|"` followed by the RGB bytes of the decoded
image.

## A note on reference captions in Spica

In `spica_train.csv`, 38,670 rows store the empty reference list as the string `"'[]'"`. `clean`
leaves them as released. For our own trainings these rows were given an empty list
(`analysis/metrics/prep_pearl_data.py`).
