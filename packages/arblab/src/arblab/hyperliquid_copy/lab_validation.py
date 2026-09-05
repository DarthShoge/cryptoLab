"""Explicitly public validation issues; never wrap raw parser exception text."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str
    required: str | None = None
    available: str | None = None


class LabValidationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))
