# Observability

## Logging

All commands emit structured JSONL logs when `--json` is enabled:

```python
@dataclass(frozen=True)
class LogEvent:
    schema_version: str
    event: str
    level: Literal["debug", "info", "warning", "error"]
    run_id: str
    artifact_id: str | None
    step: str
    message: str
    fields: Mapping[str, Any]
```

Human-readable logs are allowed by default, but every release gate consumes JSONL
or report JSON.

## Metrics

Dataset metrics:

- source row count;
- parse success rate;
- filter counts by reason;
- dedup rejection rate;
- split counts by repo and source;
- token length percentiles;
- edit size percentiles;
- license counts.

Training metrics:

- `loss/total`;
- `loss/prediction_mse`;
- `loss/sigreg`;
- `loss/sigreg_weighted`;
- optional retrieval loss;
- embedding effective rank;
- embedding effective rank ratio;
- per-dimension variance min/median/max;
- mean pairwise cosine;
- embedding norm mean/std;
- nearest-neighbor entropy;
- gradient norm;
- throughput examples per second;
- peak memory.

Evaluation metrics:

- `Recall@1`, `Recall@5`, `Recall@10`;
- MRR;
- median rank;
- patch-surprise AUC;
- baseline deltas;
- per-source and per-edit-size slices.

## Artifact Lineage

Every dataset, checkpoint, index, and report includes:

- source git SHA;
- command argv;
- config hash;
- parent artifact IDs;
- file checksums;
- schema version;
- created timestamp in UTC;
- host platform and Python version;
- dependency lock hash when available.

## Redaction Rules

Logs and reports must not include:

- secrets;
- raw environment variable values;
- full source code snippets by default;
- absolute user home paths by default;
- private repository names unless the input source is explicitly marked public.

## Kill Reports

Training and evaluation gates write a kill report when they fail. The report must
name the failed threshold, observed value, command, config hash, and suggested
next RFC-governed action.

Collapse kill reports use `schema_version=codelewm.eval.kill_report.v1` and
include:

- `reason=embedding_collapse`;
- `collapse_report` with effective-rank, variance, cosine, norm, and
  nearest-neighbor entropy metrics;
- one entry per failed threshold;
- command and config hash when available.
