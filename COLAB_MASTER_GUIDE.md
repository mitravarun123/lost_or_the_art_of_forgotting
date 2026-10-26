# Run the complete experiment in Colab

Use the two updated deliverables together:

1. `notebooks/MASTER_Colab_Whole_Experiment.ipynb`
2. `lost-or-inaccessible.zip` (in the project’s parent directory)

Open Google Colab in your own signed-in browser and choose Upload notebook. Select the master `.ipynb`. Choose a GPU under Runtime → Change runtime type if available. Choose Runtime → Run all. Authorize your Google Drive and upload the project ZIP when the notebook requests it. Do not upload the ZIP as the notebook itself.

The master notebook installs needed dependencies, extracts/reuses the source ZIP, runs software preflight tests, verifies/downloads CIFAR-10, prints the full study configuration, and runs E1–E9. E10 then renders the measured figure families and tables. It runs the full five-seed profile with all configured budgets, repeated controls, linear/nonlinear probes and restoration. It never silently substitutes a pilot or synthetic data.

Persistent Drive folder: `MyDrive/lost-or-inaccessible/`.

- `datasets/`: verified official CIFAR-10.
- `uploads/`: source ZIP, named by checksum, plus the resume pointer.
- `results/colab_<fingerprint>/`: actual measured artifacts and stage checkpoints.
- `exports/`: PNG-only figure ZIP and measured-report ZIP.
- `colab_pipeline_latest.log`: execution log appended across sessions.

**Figures are PNG only, at 600 DPI by default.** The PNG ZIP contains no PDF files. Panels have explicit titles, target classes, layers, axis labels, legends outside data panels and explanatory captions. The notebook gallery uses smaller preview thumbnails; the files in Drive and the PNG ZIP remain full-resolution. A boundary audit and pixel dimensions are recorded in `figures/figure_manifest.json`. Visual inspection of final measured figures is still necessary.

There are 5,550 configured neuron/edge intervention conditions and 22,230 configured probe fits across five seeds, including explicitly labeled readout-reference probes. This full study may need multiple runtime sessions. It is not reasonable to guarantee a one-session finish. To resume, reopen the same notebook, use the same Drive account, keep `UPLOAD_NEW_ZIP=False`, and rerun. Saved model epochs, completed stages, conditions and probe fits are reused. A source/configuration change intentionally creates another run.

The final cells save both bundles to Drive and request browser downloads. If your browser blocks multiple downloads, use the ZIPs already in the Drive `exports` folder. To regenerate figures only, run setup/source cells and then report/gallery/export cells; the latest run path is read from `LAST_RUN.txt` and no model retraining is required.

The completion grid must show every seed’s E1–E8 complete and the E9 marker before the notebook labels the whole study complete. Missing figure families stay missing rather than receiving fabricated values. F12 needs independent model replications. All actual conclusions must be based on executed research results.

Local validation covers software fixture execution, notebook code syntax, report-only execution without data loading, high-resolution PNG output, and cross-seed figure rendering. The notebook has not been executed inside your authenticated Colab account. No CIFAR-10 research results are included in this delivery.
