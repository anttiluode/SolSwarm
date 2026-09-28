# SolSwarm

**Can a world remember what happened strongly enough that a linear operator reveals what the population is becoming before the population visibly gets there?**

SolSwarm is a deliberately small falsifier built from four familiar ingredients:

- **stigmergy:** persistent local traces alter later opportunities;
- **positive-operator dynamics:** a population evolves by normalized multiplication;
- **importance sampling:** nonuniform scouts are corrected back to the intended world distribution;
- **Oja's rule:** an online stochastic eigensolver extracts a dominant mode from the stream.

The point is not to claim a new swarm optimizer. The point is to separate what actually survives measurement.

## Gate A — the world writes the future operator

A ten-state toy world has two route families, A and B. Persistent trace `s` modifies a positive operator

\[
L(s)=M\,\mathrm{diag}(\exp(b+\alpha s)),
\qquad
q_{t+1}=\frac{L(s)q_t}{\mathbf 1^\top L(s)q_t}.
\]

Two counterfactual worlds start from the **same population** and receive the **same total trace** in B. The only difference is the address of that write.

Committed v0 receipt:

| matched write | B/A spectral growth | B mass after 20 steps |
|---|---:|---:|
| B hub | **1.0723** | **0.6558** |
| B leaf | **0.9749** | **0.4346** |

At the fork, B mass is `0.5` in both worlds. The spectral ordering has already changed before the visible population separates.

The finite-difference susceptibility of `log(rho_B/rho_A)` is also address-sensitive: the B hub is about `0.947`, while each B leaf is about `0.263`. Same amount of history; different structural leverage.

## Gate B — biased scouts do not have to bias the learned mode

The second control has eight sites with 2-D feature vectors. Under the real world distribution `p` (uniform), the batch principal direction is approximately

```text
[0.9814, 0.1921]
```

A deliberately misleading "error" field makes scouts sample from `q`, placing **83.2%** of their visits on two high-error vertical sites.

Naive Oja therefore learns the **proposal's** mode, not the world's mode.

Importance-weighted Oja uses

\[
w_i=\frac{p_i}{q_i}
\]

inside the update. Across 32 deterministic seeds and 3,000 observations:

| estimator | median angle from true world PC |
|---|---:|
| naive biased Oja | **74.37°** |
| importance-weighted Oja | **2.25°** |

The worst weighted run is `5.97°`.

So the useful claim is narrow:

> scouts may look nonuniformly while an importance-corrected online eigensolver still estimates the mode of the underlying world distribution.

## Gate C — correction is not optimization

The third gate closes the loop. A site's stigmergic trace grows when its raw Oja update is large; the trace biases later scout visits, while 20% uniform exploration keeps every site reachable.

At 200 observations across 64 seeds:

| sampler | median angle |
|---|---:|
| uniform | **3.25°** |
| adaptive + importance correction | **7.49°** |
| adaptive naive | **17.44°** |

The corrected adaptive sampler is much less biased than the naive one, but **it does not beat uniform sampling** at equal budget (`29.7%` seed-wise wins).

That negative result matters. Importance correction can restore the intended statistic; it does **not** guarantee that the proposal has lower variance or learns faster.

This is the same warning raised by the BiomorphicSwarm experiments: "look where error is high" is not automatically "look where another observation is useful."

## What this connects

SolSwarm's positive branch is the operator picture:

\[
\text{local history}
\rightarrow
s
\rightarrow
L(s)
\rightarrow
\text{spectral ordering}
\rightarrow
\text{later macroscopic population}.
\]

Its sampling branch is:

\[
x_t\sim q_t
\rightarrow
\frac{p(x_t)}{q_t(x_t)}\,\text{evidence}
\rightarrow
\text{Oja mode estimate}.
\]

Together they give a concrete question for larger systems:

> can local, noisy, deliberately biased observations recover the latent modes of a persistent world early enough to forecast a collective transition before ordinary state variables make it obvious?

That is still open. SolSwarm v0 earns only the two small pieces above and keeps the failed adaptive-speedup gate visible.

## Run

```bash
python -m venv .venv
# activate it
pip install -e ".[test]"
python experiments/run_all.py
pytest
```

The deterministic receipt is committed at [`receipts/latest.json`](receipts/latest.json).

The repository root also contains a dependency-free `index.html` for GitHub Pages.

## Layout

```text
src/solswarm/operator.py   world-written positive operator + susceptibility
src/solswarm/oja.py        biased scouts + importance-weighted Oja
src/solswarm/adaptive.py   closed-loop stigmergic proposal boundary
experiments/run_all.py     regenerate authoritative receipts
receipts/latest.json       frozen v0 numbers
index.html                 explanatory public demo
tests/                     scientific invariants and claim boundaries
```

## Claim boundary

SolSwarm does **not** establish new physics, biological equivalence, optimal active sensing, a universal swarm advantage, or a new PCA algorithm. Positive operators, importance sampling, Oja/PCA, diffusion-like exploration, and stigmergy all have established literatures. The experiment is about putting them in one falsifiable loop and keeping the boundaries between them explicit.

## Pointers

- E. Oja, *Simplified neuron model as a principal component analyzer*, Journal of Mathematical Biology 15 (1982).
- A. Katharopoulos & F. Fleuret, *Not All Samples Are Created Equal: Deep Learning with Importance Sampling*, ICML 2018.
- S. Pal, F. Y. Wang & M. J. Buehler, *SwarmWorld: Stigmergic technological evolution in societies of language-model agents*, arXiv:2608.26081 (2026).
