# SolSwarm v0 Design

## Purpose

SolSwarm tests one narrow synthesis: a population can leave persistent local traces that alter a positive future-search operator; biased scouts can deliberately sample that world without changing the target statistics when their observations are importance-corrected; and an Oja-like online learner can recover a dominant mode from that biased stream.

The project must distinguish three claims instead of treating them as one:

1. **Addressed stigmergy can change macroscopic future modes.**
2. **Importance correction can remove the statistical bias introduced by nonuniform scouts.**
3. **Adaptive usefulness-chasing is not automatically better than random/uniform sampling.**

A negative result on (3) is a valid v0 result.

## Scientific gates

### Gate A — same amount of history, different address

Use a ten-state positive system split into two five-state route families A and B. Each family is a star-like local transport module. A has a small baseline growth advantage. The state evolves by normalized positive dynamics

\[
q_{t+1}=\frac{L(s)q_t}{\mathbf{1}^{\top}L(s)q_t},
\qquad
L(s)=M\,\mathrm{diag}\!\left(\exp(b+\alpha s)\right).
\]

Two counterfactual worlds start from the same `q0` and receive the same scalar write into family B. One write lands at B's hub and the other at a B leaf. The gate passes when the hub write flips the B/A within-module spectral-growth ratio above 1 while the leaf write does not, and later B mass separates although current population and total write were identical at the fork.

Also report local spectral susceptibility using finite differences of `log(rho_B/rho_A)` with respect to each trace coordinate.

### Gate B — biased scouts + importance-weighted Oja

Use a fixed eight-site, two-dimensional feature world. The uniform target distribution `p` has a known principal covariance direction. An "error" field deliberately biases a proposal distribution `q` toward two vertical high-error sites, so naive Oja learns the proposal's mode instead of the world's mode.

Run the same biased samples in two estimators:

- naive Oja;
- importance-weighted Oja with weight `p_i/q_i`.

Across deterministic seeds, the weighted estimator must recover the batch principal component while naive Oja remains far away.

### Gate C — closed-loop boundary

Let an adaptive stigmergic trace be updated by the magnitude of the local raw Oja update. Scouts sample from a mixture of a trace-softmax proposal and uniform exploration. Compare:

- uniform Oja;
- adaptive naive Oja;
- adaptive importance-weighted Oja.

This gate is a boundary, not a required "win." v0 must report whether importance correction restores the target mode and separately whether the adaptive proposal actually beats uniform sampling at the same observation budget. If it does not, preserve that negative result.

## Public demo

`index.html` is dependency-free and must work under the existing static GitHub Pages workflow. It should expose the two earned ideas visually:

- matched hub/leaf writes with live B-mass trajectories and the spectral ratio visible before the population separates;
- biased scout sampling with naive vs importance-corrected Oja vectors relative to the target principal direction.

The page is explanatory, not a second scientific implementation. Python receipts are authoritative.

## Reproducibility and claim boundaries

- Python 3.10+.
- NumPy is the only runtime dependency.
- Pytest is the test dependency.
- All random experiments use explicit seeds.
- Deterministic receipt JSON is committed under `receipts/latest.json`.
- No claim of new physics, biological equivalence, optimal sampling, or general swarm superiority.
- Error-chasing remains a control; v0 does not assume it is useful.
