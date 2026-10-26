# Lost or Inaccessible?

A PyTorch research pipeline for comparing neuron ablation, connection ablation, post-intervention decodability, and computation restoration in a **3072 → 512 → 256 → 128 → 2 ReLU MLP**, using only CIFAR-10 cats and dogs.

**Status:** Implementation and software verification are distinct from research validation. Read [EXECUTION_REPORT.md](EXECUTION_REPORT.md) for what actually ran. The manuscript is a protocol draft; full five-seed findings are pending. No classification collapse or failed probe is treated as proof of erasure.

## Setup

Python 3.10+ is required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run.py --smoke-test --device cpu
```

PyTorch/torchvision must be compatible; use official installation instructions if your existing versions conflict. All required libraries were already present in the development environment, so no global package changes were necessary. Tests use synthetic tensors only as software fixtures, never as experimental evidence.

## Run on Mac M1

```bash
python run.py --profile mac --experiment all --download
python run.py --profile mac --experiment e1 --download
python run.py --profile mac --experiment e4
python run.py --profile mac --experiment e7 --device cpu
```

Default device order is CUDA → MPS → CPU. `--device cpu|mps|cuda` overrides it. MPS's supported CPU operation fallback is enabled before torch imports; if an unsupported operation still fails, rerun the same command with `--device cpu`. Saved tensors are portable CPU-loaded artifacts. Results from different devices need not be bitwise equal. The Mac pilot uses one seed, batch size 64, four analysis batches for circuit discovery, two random repetitions, two budgets per intervention family, and linear probes. E6 is also generated, while E8/E9 can be requested explicitly. No memory capacity is assumed.

## Run on Colab

Upload the repository directory (or its ZIP and extract it), open `notebooks/colab_full_pipeline.ipynb`, set `PROJECT_DIR`, and run the cells. Drive mounting is optional; use its output path to persist checkpoints. No paid service or API is used.

```bash
python run.py --profile colab --experiment all --download
python run.py --profile colab --experiment e7
```

The full profile trains five independent models and probes every intervention condition with both probe families. This is a substantial experiment, especially all-condition nonlinear probing; Colab runtime limits can interrupt it. Resume with the identical code, profile, override file, and output path. Individual conditions, probe fits, completed stages, and training epochs are checkpointed. Full completion is not assumed from starting a notebook.

A one-epoch **execution check**, using real CIFAR-10 when accessible:

```bash
python run.py --profile mac --config configs/short_check.yaml --experiment e1 --device cpu --download
```

Overrides are merged into the base and profile YAML. To change seeds, budgets, probe epochs, random repetitions, or batch sizes, write an override YAML and pass `--config path.yaml`. Keep the same override for subsequent stages. Altering configuration creates a separate run; it does not overwrite the old one.

## Experiment dependencies and artifacts

`all` follows the profile's declared experiment list and recursively resolves `configs/experiments.yaml`. Asking for E7 also runs or reuses its baseline, ranking, and intervention dependencies. E9 resolves restoration too. E10 creates plots and tables. The runner reuses successful stages and condition/probe artifacts. Failed stages emit `.failed.json` diagnostics and stop; data-loading failures appear in the command log. A failed MPS operation is not silently replaced by synthetic outputs.

Runs live at `results/<profile>_<config-and-source-hash>/`. Each includes `config.json`, `environment.json`, `splits.json`, and per-seed directories. Per-seed artifacts include:

- `checkpoints/best.pt`, resumable `last.pt`, and training history.
- `e1.csv` … `e8.csv` for executed stages; E2 neuron rankings and dense edge score tensors.
- `masks/`, `circuits/`, `conditions/`, and `.done.json` stage records.
- `predictions/` with test labels and logits in original deterministic order.
- `probes/` fit parameters and normalization; `probe_metrics/` completion records.

Aggregate files include `robustness.csv`, `paired_tests.csv`, `paired_image_bootstrap.csv`, and `circuit_stability.csv` when E9 runs. Figures F1–F12 are generated only when their inputs exist, as 600-DPI PNG only. One seed cannot produce a cross-seed confidence-interval plot. T1–T10 exports include measured CSVs and LaTeX summaries. Full records are in CSV; LaTeX summaries may pool conditions and should be narrowed for a final paper. Derived tables/plots can be regenerated; measured completed stage artifacts are reused. Source changes intentionally create a different run fingerprint.

E1 evaluates the validation-selected checkpoint. E2 uses analysis labels for ranks. E3 includes two target classes and five rank/control methods. E4 streams edge scores, checks graph connectivity, and intervenes on retained/complement supports. E5 includes global and four layer-specific masks. E6 records comparisons by affected connections, weight magnitude, location, and representation change; performance matching uses validation scores. E7 fits fresh held-out probes at affected and subsequent layers. E8 restores predeclared edges or same-input clean activations. E9 reports seed-level inference separately from image bootstraps. See [methodology](docs/METHODOLOGY.md) for exact estimands and caveats.

## Manuscript

`paper/main.tex` is a complete conventional research-protocol manuscript with hypotheses, equations, planned result sections, limitations, and verified references. It is **not an experimentally validated paper**. Copy measured outputs from a completed run:

```bash
python scripts/export_paper.py results/mac_RUN_HASH
```

Then inspect `paper/EXPORT_SOURCE.json`, update Results/Abstract/Conclusion based only on those measured records, and compile in the built-in LaTeX editor or with `pdflatex main; bibtex main; pdflatex main; pdflatex main` from `paper/`. Publication readiness still requires full execution, author information, figure selection, literature review, and scientific review. References and their verified primary records are listed in `docs/BIBLIOGRAPHY_AUDIT.md`.

## Troubleshooting

If download fails, check internet access to torchvision's official CIFAR mirror, rerun `--download`, or place the official extracted `cifar-10-batches-py` under `--data-root`. Its checksums are validated by torchvision. Do not substitute another dataset. If baseline performance is poor, inspect class counts, normalization, training/validation losses, selected checkpoint epoch, and confusion matrices. Diagnose using training/validation/analysis data; leave test outcomes out of tuning. Report remaining poor performance honestly. One epoch is not a baseline convergence test.

For memory pressure, reduce `batch_size` and `analysis_batches`; reduce intervention budgets/repetitions for a separately labeled pilot. For MPS errors use CPU. For package errors use a fresh virtual environment. For interrupted training, `last.pt` retains optimizer, epoch, RNG and best-loss state; minibatch shuffles are independently seeded by epoch. CUDA RNGs are saved; MPS exact continuation is not guaranteed across software/hardware versions. Do not load untrusted PyTorch checkpoints.

## Persist the dataset in Google Drive

Open `notebooks/colab_download_to_drive.ipynb` in Colab and run its cells. Sign in to your intended Google account and authorize the Drive mount yourself. It downloads official CIFAR-10 into `MyDrive/lost-or-inaccessible/datasets`, verifies archive/file checksums and the 10,000/2,000 cat/dog counts, then writes `DATASET_VERIFIED.json` only on success. No research training is started. This download notebook has been syntax-checked locally, but has not run in an authenticated Colab session.

The full-pipeline notebook now defaults to `USE_DRIVE = True` and reads that same persistent dataset folder, with results saved alongside it under `MyDrive/lost-or-inaccessible/results`. Set `USE_DRIVE = False` to use temporary runtime storage instead. The full original archive is about 170 MB; the original extracted dataset is also retained.

## Master Colab notebook — updated 26 October 2026

Use **`notebooks/MASTER_Colab_Whole_Experiment.ipynb`** together with the updated **`lost-or-inaccessible.zip`** supplied alongside it. Upload the notebook to Colab, select a GPU if available, and choose Runtime → Run all. Authorize Drive and select the ZIP when prompted. The source ZIP is saved in Drive and reused automatically on subsequent sessions. The notebook runs the full five-seed profile, including all-condition linear/nonlinear probes, restoration and statistics. This is a large study; completion may span sessions. Configuration is printed and the full default is never silently reduced to a pilot.

The notebook verifies stage completion, then renders **PNG only at 600 DPI**, displays lightweight preview thumbnails, and saves/downloads a PNG-only ZIP plus a measured-report ZIP. Original checkpoint and probe artifacts stay in Drive. Figures distinguish target class, intervention layer and probe family. Shared legends sit outside panels, captions describe uncertainty, and the `figure_manifest.json` records image dimensions/DPI and a boundary check. The PNG-only bundle contains no PDFs. The older full-pipeline notebook filename now contains the same master workflow.

The source archive excludes datasets, prior results, logs, caches, and PDFs. Rebuild it with `python scripts/package_project.py`. The figure generators do not fabricate missing families: F12 requires independent seed replications, and full study completion requires all configured seeds/stages. The schematic F1 contains no empirical data.

To regenerate figures from measured results without loading data or rerunning training:

```bash
python run.py --report-only results/colab_RUN_HASH --figure-dpi 600
```

`--skip-report` runs requested experiments without rendering figures; the master notebook uses this and renders E10 once afterward. `LOI_FIGURE_DPI` can reduce software-test exports only; final research PNGs default to 600 DPI. Older documentation describing PDF figure exports is superseded by this PNG-only update.

## Revised one-hour-session pilot

Use `notebooks/REVISED_MASTER_Colab_Pilot.ipynb` with the matching revised project ZIP, rather than the original full-study notebook. See `REVISED_COLAB_GUIDE.md` for its fixed scope: seed 42, final-hidden and outgoing-weight interventions, two budgets, three random repetitions, 32 conditions, 33 linear-probe fits, same-input activation restoration, and eight 600-DPI PNG figure families. A 45-minute experiment guard preserves completed artifacts for resume; setup/download/reporting mean one-hour completion is not guaranteed. The original full study remains available separately.
# lost_or_the_art_of_forgotting
