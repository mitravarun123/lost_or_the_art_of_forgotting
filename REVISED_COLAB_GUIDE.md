# Revised one-seed Colab pilot

Use `notebooks/REVISED_MASTER_Colab_Pilot.ipynb` with the matching `lost-or-inaccessible-revised.zip` in the project’s parent folder. Upload the notebook to Google Colab, select a GPU if available, then choose Runtime → Run all. Authorize your Drive account and upload the revised project ZIP when prompted. Upload the notebook file directly; do not copy HTML/Markdown-formatted chat code into a Python cell.

The original complete study remains available in its separate notebook/profile. This revised notebook explicitly executes `--profile colab --config configs/revised.yaml`; it cannot accidentally run the old full configuration.

## Executed pilot scope

- Original CIFAR-10 Cat/Dog data, fixed 8,000/1,000/1,000 training/validation/analysis split, and 2,000 official held-out test examples.
- Original 3072→512→256→128→2 ReLU MLP; seed 42.
- Baseline cap 30 epochs, patience 5. These smaller caps are disclosed rather than claiming convergence.
- Final-hidden neuron ablation and outgoing/output-weight ablation only.
- Budgets 10% and 40%, chosen before test evaluation.
- Class-selective neurons / high-attribution edges versus three uniform-random repetitions per target/budget.
- 16 neuron and 16 edge conditions: 32 total.
- Final-hidden linear probes: one clean baseline plus one fit per condition, 33 total. Maximum probe epochs 20, patience 4.
- Output-edge ablation does not change hidden states. Its probes are explicitly labeled unaffected hidden-state references; they are excluded from the modified-state recoverability scatter.
- Same-input activation restoration for all 16 neuron-mask conditions. Full final-hidden patching recovers the original computation by construction and is a restoration sanity check, not evidence for a unique circuit.
- Within-model paired image bootstrap summaries; no cross-seed inference, standard deviation, or model-population confidence intervals from one trained model.
- Streaming edge attribution on analysis examples; no comprehensive connected-circuit extraction/search in the revised scope.

## Time limit and resume

The experiment cell has an explicit 45-minute wall-clock guard. Setup, data acquisition, tests and reporting also consume time, so a one-hour end-to-end finish is not guaranteed. If the guard is reached, it stops the worker, preserves already completed artifacts, and proceeds to report available results with incomplete stages labeled honestly. A partially completed training epoch may need to rerun; checkpoints are saved at each completed epoch. Interrupted conditions/probes resume from their last completed artifact.

Rerun the identical revised notebook and source ZIP with the same Drive account, leaving `UPLOAD_NEW_ZIP=False`. Source/config fingerprints keep versions separate. This pilot uses `MyDrive/lost-or-inaccessible/revised_results/`, `LAST_REVISED_PROJECT_ZIP.txt`, `LAST_REVISED_RUN.txt`, and `revised_pipeline_latest.log`, independently of the old full-study run pointers.

## Eight PNG figures

P1 training curves; P2 baseline confusion matrix/confidence; P3 neuron selectivity; P4 neuron ablation; P5 edge ablation; P6 probe and readout performance; P7 functional loss versus modified-hidden-state decodability; P8 activation restoration. All are 600-DPI PNGs with labeled panels and outside legends. Missing measured inputs produce missing figures, never fabricated placeholders. The gallery displays smaller previews while exports retain full resolution. The notebook saves and downloads a PNG-only ZIP and a separate measured-report bundle. Checkpoint/probe files remain in Drive.

## Validation boundary

The revised pipeline and eight-figure export are tested on explicit synthetic software fixtures. This is not CIFAR-10 research evidence. No real experiment completion or one-hour hardware performance is claimed until the notebook runs in your Colab account. Poor baseline performance and negative/ambiguous intervention outcomes must be reported as measured.
