# CodeLeWM Preliminary Results Announcement

This document is reusable public copy for announcing the current CodeLeWM
boundary. It is intentionally claim-safe.

## Short Announcement

CodeLeWM has its first public artifact-backed result. The infrastructure works:
we can build public-safe Python edit datasets, train code-edit transition models
on HF Jobs, publish dataset/model/results artifacts to Hugging Face, download
them again, and rerun verification locally.

The scientific result is negative. The tested action-conditioned objectives do
not beat the no-action baseline, latent probes do not support semantic-axis
claims, and downstream patch-ranking usefulness is not established yet.

That is still useful: the harness is now ready for the next falsifiable test,
where an LLM proposes candidate patches and CodeLeWM scores or reranks them.

## X / Twitter Draft

CodeLeWM has its first public artifact-backed result.

The pipeline works: HF Jobs training, public HF dataset/model/results artifacts,
clean downloads, manifest checks, eval reruns, and secret scans.

The science is negative so far: action-conditioned variants still lose to
the no-action baseline, latent-axis claims are unsupported, and downstream
usefulness is not proved yet.

Next test: LLM generates candidate patches, CodeLeWM scores/reranks them.

## Longer Post

CodeLeWM is a research harness for learning latent transition models over code
edits. The goal is not to generate code directly. The useful target is to score
candidate transitions: given a before-state, an edit instruction, and candidate
after-states, can a learned world model rank the candidate that best matches the
requested change?

The current milestone validates the artifact pipeline:

- public-safe dataset construction;
- HF Jobs training;
- public Hugging Face dataset/model/results artifacts;
- clean `hf download` verification;
- retrieval, surprise, ablation, scorer-quality, score, and rerank checks;
- model cards, dataset cards, and benchmark reports.

The current model-quality result is negative. Text-action scoring does not beat
the no-action baseline on the tested scaled runs, including the v0.2
action-swap/inverse-action intervention. The latent-probe gate is unsupported,
and the downstream scorer-quality path is still a one-example smoke test.

That makes the next milestone clear. The next public use case is an LLM +
world-model harness: use an LLM through OpenRouter to propose multiple candidate
patches, store them as untrusted candidate packs, and ask CodeLeWM to rerank
them against no-action and LLM-order baselines.

The artifact index is `docs/benchmark/PUBLIC_ARTIFACT_INDEX_2026-05-21.md`.
The detailed preliminary report is
`docs/benchmark/PRELIMINARY_RESULTS_2026-05-21.md`.

## Do Not Say

- CodeLeWM improves coding.
- CodeLeWM has validated high-level semantic latent dimensions.
- The v0.2 checkpoint beats no-action.
- The downstream reranker is proved useful.

Those claims remain blocked until the downstream benchmark gate passes.
