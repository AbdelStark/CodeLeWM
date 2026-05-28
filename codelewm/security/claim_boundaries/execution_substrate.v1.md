# CodeLeWM Execution-Substrate Claim Boundary (v1)

This text scopes what claims an artifact derived from the execution-trace
substrate (`codelewm.data.execution_pack.v1` and the v0.6 model line) is
allowed to make. It is embedded verbatim into dataset cards, model cards,
and JSON manifests. The SHA-256 of this file is recorded in every artifact
that depends on the substrate.

## What This Substrate Is

The execution-substrate artifacts are derived from running licensed public
Python submissions in a sandbox at data-build time. The sandbox runs each
`(code, input)` pair in an isolated subprocess under a stdlib-only import
policy, with network denied, filesystem writes audited, CPU and memory
limited, and outputs gated by a determinism re-run.

The resulting data artifact contains tokenized code, tokenized inputs,
tokenized outputs, and metadata. It contains no executable payload.

## Where Code Runs

| Phase | Executes user code? |
|-------|---------------------|
| Data-build (sandboxed) | Yes — once per `(code, input)` to capture the output, then a second time to validate determinism. Results are stored as data only. |
| Training | No. The training path consumes tokenized data through the same parsing and tokenization helpers used for the commit-edit substrate. |
| Inference / scoring / reranking | No. The scorer and reranker operate over latents and tokenized inputs only. |
| Demo (`bugfix-edge-case` scenario) | No. |
| Demo (`execution-rerank` scenario) | Yes — only to label LLM-sampled completion correctness against hidden tests, on operator-reviewed problem sets, using the same sandbox under the same policy. |

The non-execution policy in `docs/spec/06-security.md` continues to govern
training, scoring, indexing, evaluation, and dataset construction. The
sandbox is the named follow-on subsystem the policy anticipates, with its
own isolation, manifest, and logging contract.

## What Claims This Substrate Can Support

Subject to passing the claim gates listed in the v0.6 run manifest and in
the issue tracker (#259), an artifact derived from this substrate may
support:

- claims about deterministic Python programs executed under the
  stdlib-only sandbox policy;
- claims scoped to the source datasets and license set the manifest
  enumerates (initially CodeNet, MBPP, MBPP-Plus, APPS, HumanEval, all
  permissive licenses);
- claims supported by per-gate evidence with bootstrap confidence
  intervals and at least two training seeds;
- comparison claims between this substrate and the v0.2 commit-edit
  substrate, as long as both reports are cited.

## What Claims This Substrate Cannot Support

This substrate does not support:

- claims about Python programs that depend on the filesystem, network,
  environment variables, third-party packages, time, randomness without a
  fixed seed, or any other non-deterministic source;
- claims about non-Python code in any language;
- claims about code requiring imports outside the stdlib allowlist;
- claims about model behavior on adversarial or pathological inputs not
  covered by the source datasets;
- claims that generalize beyond the source-dataset license envelope.

## Threat Model

The sandbox treats the source code in each `(code, input)` record as
**untrusted**. The build host is the trust boundary: the sandbox isolates
the subprocess but does not attempt to defeat a determined attacker who
controls the source code. This is acceptable because:

- inputs to the sandbox are licensed public Python submissions from
  well-known datasets;
- the sandbox runs on a controlled build host, not on a user's machine
  during training or inference;
- the output of the sandbox is a data artifact, not an executable;
- the artifact is published only after manifest verification and secret
  scanning.

Operators who reuse the sandbox in the `execution-rerank` evaluation
scenario must run on a controlled host and treat LLM-sampled completions
as untrusted with the same threat model.

## Required Card Language

Both the dataset card and the model card MUST include the following
paragraph verbatim:

> The execution-pack data artifact is the deterministic output of running
> licensed public Python submissions in an isolated sandbox under a
> stdlib-only policy at data-build time. The artifact contains no
> executable payload; it contains tokenized code, tokenized inputs,
> tokenized outputs, and metadata. Training and inference never execute
> code. The sandbox is reused only in the dedicated downstream-evaluation
> scenario (`execution-rerank`) to label completion correctness against
> hidden tests, and only on inputs the operator has reviewed.

## Versioning

This file is versioned by filename suffix (`execution_substrate.v1.md`).
Any change to the boundary text increments the suffix and emits a new
fingerprint. Manifests that reference the boundary record the exact
filename and the SHA-256 fingerprint so consumers can detect drift.
