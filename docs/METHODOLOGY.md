# Methodology and interpretation contract

## Data and evaluation separation
Only CIFAR-10 cat (3→0) and dog (5→1) are used. A fixed split seed 314159 stratifies the official training subset into 8,000 model training / 1,000 validation / 1,000 analysis. Official test data have their own 2,000 IDs. The manifest records original torchvision indices, normalization moments, and a SHA-256 of the training labels. All ranks and masks use analysis data; model selection uses validation loss. No test outcomes are used for candidate selection or matching.

Probe fit/validation partitions are a stratified 6,400/1,600 subdivision of model-training data. This uses data already seen by the frozen base model, which must be disclosed. No official test sample fits or selects a probe. Evaluation is the same official test set to support paired outcome comparison; test results must not motivate further tuning. To perform exploratory test-informed work, label it exploratory and obtain new evaluation data before confirmatory claims (outside this primary implementation).

## Neurons and edges
Selectivity S is mean(cat) − mean(dog). Cat ranking uses descending S, dog uses descending −S. Low controls reverse that ranking. Gradient ranks use mean absolute activation-times-gradient of cat–dog margin; the absolute margin attribution is identical for the negated binary margin. ROC-AUC uses dog as positive and stores the signed-direction AUC and max(AUC,1−AUC) separately. Matching samples within 20 empirical magnitude quantile bins, approximately matching the selected support's activation/weight magnitudes. Matching may overlap targeted indices; it is a constrained random control, not a disjoint control.

Edge scores use E[g_j W_ji a_i], E[|g_j W_ji a_i|], and their class-conditional versions. Gradients target the sum of per-example cat–dog margins, so there is no minibatch mean-gradient scaling error. Use GᵀA products, retaining O(sum of layer edge counts) scores and only a minibatch of activations. The Mac cap uses a deterministic first four shuffled analysis minibatches (recorded counts); all analysis examples contribute to neuron ranks. Absolute edge scores establish candidates, not causal circuits.

Top-ranked supports can be disconnected. A second extractor builds complete maximum-score paths via dynamic programming and takes their union, processing both output targets. Each target's paths share an output, but the two unions need not share nodes. Report explicit connectivity instead of assuming it. Repeated paths may deduplicate, so actual sparsity can differ from the nominal path-count budget. Biases are not masked: the computational graph covers learned linear edges, with biases treated as persistent exogenous inputs.

## Faithfulness and completeness — defined before computation
For margin m_t(f,x)=f_t(x)−f_(1−t)(x), let D=max(E_analysis |m_t(f,x)|,1e−8). For candidate C, f_C retains C and removes every other weight edge; f_\C removes C while retaining the complement. All biases remain fixed.

F(C)=1−E_analysis |m_t(f_C,x)−m_t(f,x)| / D.

K(C)=E_analysis |m_t(f_\C,x)−m_t(f,x)| / D.

F can be negative and K can exceed 1. They measure normalized margin fidelity and sensitivity, not certified completeness or a unique minimal mechanism. Degenerate small baseline margins are reported through D. E4 validates these on analysis data; E5 also computes these quantities on the fixed final test set for high-attribution supports, without tuning candidates on their outcomes. Retention is never described as removal.

## Comparisons and recoverability
Neuron zeroing suppresses each selected neuron's outgoing connections. Incoming weights still compute a preactivation, so affected-connection counts use outgoing edges; biases are excluded from magnitude denominators. Edge masks count removed effective weights. Neither quantity directly equals computational savings. Representation perturbation is ||A'−A||_F / max(||A||_F,1e−12), at each hidden layer. Mean activation norms are also saved.

For an edge intervention on layer l, affected hidden layers begin at l. Global edge masks begin at hidden layer 0. Neuron interventions begin at their post-ReLU layer. All subsequent hidden representations are probed, including the final hidden state. Output-edge-only interventions affect no hidden state, so the final-hidden probe is explicitly `unaffected_pre_readout_reference`; do not call this post-hidden-intervention evidence.

Both probe families receive identical dimensions and train-only feature standardization. Each condition fits fresh parameters; early stopping uses probe validation loss. Linear probes are two-logit logistic regression optimized with cross entropy; nonlinear probes use width 64 and ReLU. Probe seeds are paired across conditions. The default full profile probes all intervention conditions; the Mac profile probes targeted conditions only. No retraining of the original model occurs after intervention.

Performance-matched pairs use validation balanced accuracy, matched per target and corresponding outgoing location (hidden layer l ↔ edge layer l+1), with gaps reported. Test matching is prohibited. Recovery plots should be interpreted with matching gaps and capacity controls. Failure of these probes is evidence about these decoder families, never a proof of information erasure.

## Restoration
Selective/random edge restoration restores a predeclared fraction of removed edges. Selected edges use original analysis scores. Full restoration reproduces baseline weights. Random repetitions remain nested within model. Activation patching records same-input clean hidden states and replaces a specified hidden layer; downstream masked edges can still block recovery. Readout-only masks have no subsequent hidden-layer patch location. Restoring original computations, fitting a new probe, and retraining a model are distinct procedures.

## Statistical units
Average random repetitions within each seed. Training seeds are independent model replications; images are paired within-model observations. Use seed means, sample SD, 95% Student-t intervals, paired targeted-minus-random accuracy differences and Cohen's dz. One seed has no seed-level SD, interval, or p-value. Zero-variance effects have undefined standardized effect sizes and p-values rather than invented finite values. Holm correction is applied over all computed targeted/random tests together.

Paired stratified image bootstraps resample identical example indices for clean/ablated predictions, within each class. Label these intervals **within-model**, never as five-seed uncertainty. Raw cross-seed edge Jaccard measures index support instability and is confounded by neuron permutation. Published cross-seed mechanism claims require alignment or permutation-invariant methods beyond this diagnostic.

## Reporting rules
All numeric research tables derive from saved measured CSVs. Missing results get `Not run`. One-epoch checks are execution checks, not validated baseline estimates. Schematic F1 is permitted without measured data. No other research plots are generated without measured experiment files. Synthetic tests stay under `results/software_smoke` or pytest temporary directories. No test fixture is a CIFAR-10 substitute.
