# Execution report — 17 October 2026

This repository is an implemented research pipeline and a compiled manuscript protocol. **The CIFAR-10 research study has not been executed, and the paper is not experimentally validated.**

| Category | Status | Evidence / scope |
|---|---|---|
| Implemented | E1–E9 plus E10 reporting | Shared data/model/checkpoint pipeline, Mac/Colab profiles, restoration, probes, statistical exports, F1–F12 generators, T1–T10 exports |
| Unit-tested | Passed | `python3 -m pytest -q`: **16 passed**, 14 environment/dependency warnings, 17.25 seconds; `logs/pytest.log` |
| Training/resume tested | Passed on software fixtures | Full primary MLP; validation-selected best checkpoint, optimizer/RNG resume and two-epoch history; not CIFAR-10 training |
| CPU smoke-tested | Passed | Full primary MLP on synthetic software fixtures, streaming attribution, ablations, both probe families, serialization, plot rendering; `results/software_smoke/status.json`, `logs/smoke.log` |
| Cross-module integration tested | Passed on software fixtures | E2–E9 together, random controls, retained/removal circuits, validation matching, probes, restoration, bootstrap/statistics, F1–F11 applicable exports and LaTeX tables; pytest temporary outputs, never research data |
| Manuscript compilation | Passed | Desktop compiler reported success after its first attempt timed out downloading TeX dependencies |
| Bibliographic metadata | Verified | Seven primary records documented in `docs/BIBLIOGRAPHY_AUDIT.md`; no novelty claim |
| CIFAR-10 acquisition | Incomplete; transfer stopped | Restricted attempt failed DNS resolution. Network-enabled retry reached the official torchvision Toronto mirror, but transferred only 8.7 MiB of 170,498,071 bytes in several minutes at approximately 40–50 KB/s. Interrupted rather than leaving an hour-long background transfer. See `logs/cifar_download_attempt.log` and `logs/short_baseline.log`. |
| Short CIFAR-10 baseline | Not executed | Dataset never finished downloading; no fitted CIFAR-10 model or accuracy is reported |
| Full five-seed E1–E9 | Not executed | Requires the complete dataset and sustained model/intervention/probe runs |
| MPS hardware execution | Not executed | Device selection/fallback implemented; actual M1 hardware behavior unverified |
| CUDA/Colab execution | Not executed | Notebook and portable artifacts implemented; no Colab runtime was connected |
| Five-seed intervals / F12 | Not executed | No model-training replications available; no fabricated intervals or plot |
| Measured research tables/plots | Not available | Only the schematic F1 and explicitly marked software-test artifacts exist. No synthetic values populate research outputs. |

## Software verification details

Tests check class filtering/remapping, stratified split independence, exact standard-MLP forward equivalence, deterministic evaluation, hidden-state extraction, identity and full neuron masks, effective edge masking, streamed versus explicit per-example edge attribution, candidate graph validity/disconnection, train-only probe preprocessing, full restoration and patching, reproducible initialization, checkpoint serialization, profile format compatibility, Holm correction, and training checkpoint selection/resume. The all-module integration test uses tiny fixture models to exercise experiment joins and exports without impersonating a research run. The standalone CPU smoke test additionally exercises the full 3072→512→256→128→2 architecture.

The environment had all required packages installed already. Observed versions: Python 3.10; torch 2.11.0; torchvision 0.16.1; numpy 1.26.4; scipy 1.15.3; scikit-learn 1.6.1; pandas 2.0.3; matplotlib 3.8.0; PyYAML 6.0.3; pytest 9.1.1. Torchvision warned about its optional native image extension; CIFAR-10's NumPy/PIL dataset path does not use it. Use a fresh environment with compatible torch/torchvision releases for new machines. Matplotlib used a temporary writable cache. No tests failed in the final run.

The initial manuscript compile timed out during first-use dependency downloads; a retry compiled successfully. No source compilation error remained. The built-in editor provides the PDF preview; the deliverable is the saved LaTeX project.

## Exact next commands

Run from the repository directory. Download completion is required before research stages can execute.

```bash
# Real-data one-epoch execution check; not a convergence study:
python run.py --profile mac --config configs/short_check.yaml --experiment e1 --device cpu --download

# Mac pilot, including linear recoverability probes:
python run.py --profile mac --experiment all --download

# Complete study on Colab (or another CUDA machine):
python run.py --profile colab --experiment all --download
```

Open `notebooks/colab_full_pipeline.ipynb` for dependency installation, optional Google Drive persistence, execution, and output inspection. Repeating the same command resumes the same configuration/source fingerprint. Full all-condition probing is computationally substantial and may span multiple Colab sessions.

Do not convert this protocol into an empirical paper until the corresponding stages have completed and their actual measurements have been inspected. Negative or poor results must remain in the report. Readout failure, failed probing, restored computation, and learned alternative readouts answer different questions.

## Master Colab / PNG-only release — 26 October 2026

- Added `notebooks/MASTER_Colab_Whole_Experiment.ipynb`: source ZIP upload and persistent reuse, Drive setup, dataset verification, full five-seed E1–E9 execution, completion audit, E10 PNG report, preview gallery and downloadable exports.
- Updated `colab_full_pipeline.ipynb` to the same master workflow.
- Replaced PDF figure exports with **600-DPI PNG only**. Figures are separated by target, location and probe family where needed; legends sit outside data panels, long labels/captions wrap, and independent-seed intervals are identified. A JSON manifest records dimensions, DPI and text-boundary flags.
- Added `--report-only`, `--skip-report`, `--figure-dpi`, explicit E9/E10 markers, stage progress and periodic E7 progress. Reporting can run without dataset loading/training.
- Final release verification: **20 tests passed**, 14 existing dependency warnings, 63.72 seconds (`logs/pytest_master_release.log`). The integrated fixture report produced 47 PNGs with zero text-boundary flags. A separate full-resolution export test verified 600 DPI and width above 8,000 pixels; representative layouts were inspected visually. Cross-seed F12 rendering was exercised on explicit software fixtures.
- The source package excludes datasets, prior results, caches, logs and PDFs. Its ZIP integrity and required entry points are checked when packaged.
- **Not executed:** authenticated Colab runtime and actual CIFAR-10 research study. No measured research conclusions or research PNGs are claimed in this release. Synthetic software-test figures are not included as research outputs.

## Revised one-seed pilot release

Added `configs/revised.yaml` and `notebooks/REVISED_MASTER_Colab_Pilot.ipynb`, using the existing data/model/checkpoint code with configurable intervention locations/methods. The original full-study defaults remain available. Revised scope: seed 42, final-hidden neuron and output-weight masks, 10%/40% budgets, three random repetitions, 32 ablation conditions, 33 final-hidden linear-probe fits, and same-input activation restoration. Baseline/probe caps are 30/20 epochs with validation early stopping. Eight dedicated PNG-only figure families are generated from measured inputs. Output-edge probes are labeled unaffected hidden-state references; whole-layer patching is explicitly a restoration sanity check.

Validation: all **22 software tests passed** (14 existing dependency warnings), including revised-scope integration, the 32/33 condition/fit counts, intervention-location restrictions, clean-output restoration, and exactly eight PNGs without boundary flags. The original full pipeline remains covered by its integration tests. Revised notebook code cells are syntax-checked. A 45-minute experiment-cell wall-clock guard preserves completed artifacts and supports resume; one-hour end-to-end runtime is not guaranteed. The actual CIFAR-10 pilot and authenticated Colab execution remain not run.
