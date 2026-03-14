# ETH3D DSLR Pose AUC Reference

This note records the current ETH3D DSLR pose-AUC result paths and the exact
evaluation-script usage needed to reproduce or compare them in a later task.

## Dataset and scene subset

- DSLR root: `dataset/eth3d/dslr`
- Scene subset source: `dataset/instantsfm_eth3d_dslr_partial_auc.csv`
- Metrics: relative pose `AUC@1/3/5/10`

The evaluator uses the scenes whose CSV rows have `scope=scene` unless
`--scenes ...` is passed explicitly.

## Evaluation script

Script:

```bash
python tools/eval_eth3d_pose_auc.py --help
```

Key behavior:

- `--run_instantsfm` runs InstantSfM in isolated workspaces.
- `--run_glomap` runs `colmap global_mapper`.
- The script always writes per-method CSVs and a side-by-side comparison CSV.
- The InstantSfM batch path uses an inline Python runner inside the evaluator.
  It does not rely on `python -m instantsfm.scripts.sfm`.
- Because of that, the evaluator can still toggle retriangulation through
  `config.OPTIONS["skip_retriangulation"]` even if `sfm.py` does not currently
  expose `--enable_retriangulation`.

## Current result sets

### 1. Full subset without retriangulation

Command:

```bash
python tools/eval_eth3d_pose_auc.py \
  --dslr_root dataset/eth3d/dslr \
  --scene_source_csv dataset/instantsfm_eth3d_dslr_partial_auc.csv \
  --run_instantsfm \
  --force_rerun_instantsfm \
  --instantsfm_workers 4 \
  --output_dir dataset \
  --prefix eth3d_compare_noretri
```

Reports:

- `dataset/eth3d_compare_noretri_instantsfm.csv`
- `dataset/eth3d_compare_noretri_glomap.csv`
- `dataset/eth3d_compare_noretri_compare.csv`
- `dataset/eth3d_compare_noretri_instantsfm_status.tsv`

Summary:

- InstantSfM `__all__`: `11.573312 / 17.237066 / 19.120765 / 21.266605`
- InstantSfM `__avg__`: `17.479827 / 26.529749 / 29.670385 / 33.122213`

### 2. Full subset with retriangulation

Command:

```bash
python tools/eval_eth3d_pose_auc.py \
  --dslr_root dataset/eth3d/dslr \
  --scene_source_csv dataset/instantsfm_eth3d_dslr_partial_auc.csv \
  --run_instantsfm \
  --force_rerun_instantsfm \
  --instantsfm_enable_retriangulation \
  --instantsfm_workers 4 \
  --output_dir dataset \
  --prefix eth3d_compare_retri
```

Reports:

- `dataset/eth3d_compare_retri_instantsfm.csv`
- `dataset/eth3d_compare_retri_glomap.csv`
- `dataset/eth3d_compare_retri_compare.csv`
- `dataset/eth3d_compare_retri_instantsfm_status.tsv`

Summary:

- InstantSfM `__all__`: `13.681806 / 20.009952 / 21.896028 / 23.732244`
- InstantSfM `__avg__`: `20.940999 / 30.283059 / 33.145737 / 35.884490`

### 3. Full subset with retriangulation (minimal retriangulation-fix branch)

Command:

```bash
python tools/eval_eth3d_pose_auc.py \
  --dslr_root dataset/eth3d/dslr \
  --scene_source_csv dataset/instantsfm_eth3d_dslr_partial_auc.csv \
  --run_instantsfm \
  --force_rerun_instantsfm \
  --instantsfm_enable_retriangulation \
  --instantsfm_workers 4 \
  --output_dir dataset \
  --prefix eth3d_compare_retri_minimal
```

Reports:

- `dataset/eth3d_compare_retri_minimal_instantsfm.csv`
- `dataset/eth3d_compare_retri_minimal_glomap.csv`
- `dataset/eth3d_compare_retri_minimal_compare.csv`
- `dataset/eth3d_compare_retri_minimal_instantsfm_status.tsv`

Run status:

- InstantSfM valid reconstructions: `24 / 24` scenes

Summary:

- InstantSfM `__all__`: `13.855182 / 20.223527 / 22.104155 / 23.933716`
- InstantSfM `__avg__`: `21.203007 / 30.748097 / 33.649596 / 36.415798`

Delta vs `eth3d_compare_retri_instantsfm.csv`:

- `__all__` delta: `+0.173375 / +0.213575 / +0.208128 / +0.201472`
- `__avg__` delta: `+0.262007 / +0.465038 / +0.503858 / +0.531308`

## Direct retriangulation effect

Comparing `eth3d_compare_retri_instantsfm.csv` against
`eth3d_compare_noretri_instantsfm.csv`:

- `__all__` delta: `+2.108495 / +2.772886 / +2.775263 / +2.465639`
- `__avg__` delta: `+3.461173 / +3.753310 / +3.475352 / +2.762277`

Largest `AUC@10` gains from retriangulation:

- `old_computer`: `+13.593233`
- `meadow`: `+9.464325`
- `botanical_garden`: `+8.319636`
- `relief_2`: `+7.692660`
- `courtyard`: `+7.313170`
- `lecture_room`: `+6.094806`

Smallest `AUC@10` deltas:

- `boulders`: `-0.174478`
- `playground`: `-0.070390`
- `lounge`: `-0.060241`

## How to reference these in a new task

Use the result files directly:

- Baseline without retriangulation:
  [eth3d_compare_noretri_instantsfm.csv](/workspaces/InstantSfM/dataset/eth3d_compare_noretri_instantsfm.csv)
- Variant with retriangulation:
  [eth3d_compare_retri_instantsfm.csv](/workspaces/InstantSfM/dataset/eth3d_compare_retri_instantsfm.csv)
- Minimal retriangulation-fix run:
  [eth3d_compare_retri_minimal_instantsfm.csv](/workspaces/InstantSfM/dataset/eth3d_compare_retri_minimal_instantsfm.csv)
- GLOMAP comparison for the retriangulated variant:
  [eth3d_compare_retri_compare.csv](/workspaces/InstantSfM/dataset/eth3d_compare_retri_compare.csv)
- GLOMAP comparison for the minimal retriangulation-fix run:
  [eth3d_compare_retri_minimal_compare.csv](/workspaces/InstantSfM/dataset/eth3d_compare_retri_minimal_compare.csv)

If a later task needs a fresh rerun, reuse the same `--prefix` values to
replace these files, or choose a new prefix to keep the old reports intact.
