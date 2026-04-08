"""Typed models for the PrivacyOps-X OpenEnv environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State

WorkspaceFieldName = Literal[
    "request_type",
    "verification_status",
    "jurisdiction",
    "sla_days",
    "priority",
    "routing_queue",
    "case_status",
    "retention_decision",
    "escalation_required",
]


class WorkspaceView(BaseModel):
    request_type: Literal[
        "unknown",
        "access",
        "erasure",
        "access_erasure",
        "suppression",
    ] = "unknown"
    verification_status: Literal[
        "unknown",
        "verified",
        "verification_required",
        "rejected_identity",
    ] = "unknown"
    jurisdiction: Literal["unknown", "cpra", "gdpr", "coppa", "other"] = "unknown"
    sla_days: int | None = None
    priority: Literal["low", "medium", "high", "urgent"] | None = None
    routing_queue: Literal[
        "triage",
        "fulfillment",
        "manual_privacy_review",
        "privacy_legal",
        "fraud_privacy_joint",
    ] | None = None
    case_status: Literal[
        "new",
        "pending_verification",
        "approved",
        "partially_fulfilled",
        "escalated",
        "denied",
        "closed",
    ] = "new"
    retention_decision: Literal[
        "none",
        "retain_billing",
        "retain_legal_hold",
        "partial_delete",
        "suppress_marketing",
    ] = "none"
    escalation_required: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RecordView(BaseModel):
    record_id: str
    record_type: Literal["account", "billing", "fraud", "minor_profile"]
    summary: str
    flags: list[str] = Field(default_factory=list)


class PolicyArticleView(BaseModel):
    article_id: str
    title: str
    excerpt: str


class MessageTurn(BaseModel):
    turn_id: str
    role: Literal["requester", "analyst"]
    channel: Literal["ticket", "email"] = "email"
    message: str
    fact_ids: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    reviewer: Literal["compliance", "legal", "audit"]
    severity: Literal["info", "warn", "fail"]
    code: str
    message: str


class FailureModes(BaseModel):
    hallucination: int = 0
    policy_violation: int = 0
    logic_error: int = 0
    unsafe_action: int = 0
    redundancy: int = 0
    verification_error: int = 0
    evidence_gap: int = 0
    overconfidence: int = 0
    requester_miscommunication: int = 0


class BenchmarkBreakdown(BaseModel):
    compliance_accuracy: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    reasoning_quality: float = Field(ge=0.0, le=1.0)
    efficiency_score: float = Field(ge=0.0, le=1.0)
    legal_consistency: float = Field(ge=0.0, le=1.0)
    robustness_score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    interaction_quality: float = Field(ge=0.0, le=1.0)
    confidence_calibration: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)


class PrivacyOpsAction(Action):
    action_type: Literal[
        "inspect_case",
        "open_record",
        "search_policy",
        "open_policy_article",
        "set_case_field",
        "add_internal_note",
        "draft_reply",
        "message_requester",
        "request_review",
        "self_review",
        "submit",
    ]
    target_id: str | None = None
    field_name: WorkspaceFieldName | None = None
    field_value: str | int | bool | None = None
    query: str | None = None
    content: str | None = None
    reviewer: Literal["compliance", "legal", "audit"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PrivacyOpsObservation(Observation):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    ticket_id: str
    ticket_summary: str
    workspace: WorkspaceView
    visible_records: list[RecordView] = Field(default_factory=list)
    visible_policy_articles: list[PolicyArticleView] = Field(default_factory=list)
    requester_thread: list[MessageTurn] = Field(default_factory=list)
    latest_requester_message: str | None = None
    revealed_requester_facts: list[str] = Field(default_factory=list)
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    explanation_trace: list[str] = Field(default_factory=list)
    last_action_result: str
    warning: str | None = None
    error: str | None = None
    draft_reply: str = ""
    risk_score: float = Field(ge=0.0, le=1.0)
    steps_remaining: int = Field(ge=0)
    sla_deadline: int = Field(ge=0)
    urgency_level: Literal["low", "medium", "high"] = "low"
    user_reaction_preview: Literal[
        "unknown",
        "satisfied",
        "confused",
        "escalated",
    ] = "unknown"


class PrivacyOpsState(State):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    seed: int | None = None
    variant_id: str
    case_inspected: bool = False
    workspace: WorkspaceView
    viewed_record_ids: list[str] = Field(default_factory=list)
    viewed_policy_ids: list[str] = Field(default_factory=list)
    visible_policy_ids: list[str] = Field(default_factory=list)
    requester_thread: list[MessageTurn] = Field(default_factory=list)
    last_requester_message: str | None = None
    revealed_requester_facts: list[str] = Field(default_factory=list)
    confidence_history: list[float] = Field(default_factory=list)
    review_history: list[ReviewFinding] = Field(default_factory=list)
    explanation_tags: list[str] = Field(default_factory=list)
    explanation_trace: list[str] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    audit_log: list[str] = Field(default_factory=list)
    note_history: list[str] = Field(default_factory=list)
    draft_reply: str = ""
    risk_score: float = Field(ge=0.0, le=1.0)
    sla_window_steps: int = Field(default=0, ge=0)
    sla_deadline: int = Field(default=0, ge=0)
    urgency_level: Literal["low", "medium", "high"] = "low"
    invalid_action_count: int = 0
    redundant_action_count: int = 0
    failure_modes: FailureModes = Field(default_factory=FailureModes)
    submitted: bool = False
    done: bool = False
    user_reaction: Literal["unknown", "satisfied", "confused", "escalated"] = (
        "unknown"
    )
    final_breakdown: BenchmarkBreakdown | None = None
