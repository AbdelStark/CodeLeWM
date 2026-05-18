"""Action-view policy for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from codelewm.model.transition import ActionView


ACTION_VIEW_POLICY_SCHEMA_VERSION = "codelewm.eval.action_view_policy.v1"
ReportScope = Literal["headline", "ablation", "diagnostic"]


class ActionViewPolicyError(ValueError):
    """Raised when an evaluation report violates action-view policy."""


@dataclass(frozen=True)
class ActionViewReportPolicy:
    """Action-view metadata that every evaluation report must validate."""

    action_view: ActionView
    report_scope: ReportScope
    diagnostic_upper_bound: bool = False
    schema_version: str = ACTION_VIEW_POLICY_SCHEMA_VERSION
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_view": self.action_view,
            "report_scope": self.report_scope,
            "diagnostic_upper_bound": self.diagnostic_upper_bound,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionViewReportPolicy":
        warnings = payload.get("warnings", ())
        if not isinstance(warnings, (list, tuple)):
            raise ActionViewPolicyError("warnings must be a list")
        return cls(
            schema_version=str(payload["schema_version"]),
            action_view=payload["action_view"],
            report_scope=payload["report_scope"],
            diagnostic_upper_bound=bool(payload.get("diagnostic_upper_bound", False)),
            warnings=tuple(str(warning) for warning in warnings),
        )


def build_action_view_report_policy(
    action_view: ActionView,
    *,
    report_scope: ReportScope,
) -> ActionViewReportPolicy:
    """Build and validate report metadata for an evaluation action view."""

    warnings: tuple[str, ...] = ()
    diagnostic_upper_bound = False
    if action_view == "patch":
        diagnostic_upper_bound = True
        warnings = ("patch action is leaky and may only be reported as a diagnostic upper bound",)
    policy = ActionViewReportPolicy(
        action_view=action_view,
        report_scope=report_scope,
        diagnostic_upper_bound=diagnostic_upper_bound,
        warnings=warnings,
    )
    validate_action_view_report_policy(policy)
    return policy


def validate_action_view_report_policy(policy: ActionViewReportPolicy | dict[str, Any]) -> ActionViewReportPolicy:
    """Return validated report metadata or raise on leakage-prone action views."""

    if isinstance(policy, dict):
        policy = ActionViewReportPolicy.from_dict(policy)

    if policy.schema_version != ACTION_VIEW_POLICY_SCHEMA_VERSION:
        raise ActionViewPolicyError(
            "unsupported action-view policy schema; "
            f"expected {ACTION_VIEW_POLICY_SCHEMA_VERSION!r}, got {policy.schema_version!r}"
        )
    if policy.action_view not in ("text", "abstract", "patch"):
        raise ActionViewPolicyError(f"unsupported action_view: {policy.action_view}")
    if policy.report_scope not in ("headline", "ablation", "diagnostic"):
        raise ActionViewPolicyError(f"unsupported report_scope: {policy.report_scope}")

    if policy.report_scope == "headline" and policy.action_view != "text":
        raise ActionViewPolicyError(
            "headline evaluation reports must use action_view='text'; "
            f"got {policy.action_view!r}"
        )
    if policy.action_view == "patch":
        if policy.report_scope != "diagnostic":
            raise ActionViewPolicyError("patch action reports must use report_scope='diagnostic'")
        if not policy.diagnostic_upper_bound:
            raise ActionViewPolicyError(
                "patch action reports must be tagged diagnostic_upper_bound=true"
            )
    if policy.diagnostic_upper_bound and policy.action_view != "patch":
        raise ActionViewPolicyError("only patch action reports can be diagnostic upper bounds")

    return policy
