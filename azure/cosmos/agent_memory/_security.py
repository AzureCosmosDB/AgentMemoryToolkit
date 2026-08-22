"""Trusted security context passed from host layers into AMT core."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SecurityContext(BaseModel):
    """Trusted identity and tenancy claims for core authorization decisions.

    Populated by the host (the InferencePlatform gateway in the managed path, or the
    application in the SDK/embedded path) and consumed by AMT core - AMT never
    authenticates or calls Entra/Graph itself. See ``Docs/inference-platform-auth-flow.md``.

    - ``tenant_id``  - the Entra tenant id (``tid``); the hard isolation boundary.
    - ``principal``  - ``user:<oid>`` (delegated) or ``agent:<app-sp-id>`` (app-only).
    - ``agent_id``   - ``agent:<app-sp-id>`` of the calling app; provenance + agent scope.
    - ``groups``     - the caller's Entra group/team ids (bare ids) from Microsoft Graph;
                       drives shared-scope (``team:<gid>``) membership.
    - ``roles``      - ``admin`` / ``reviewer`` etc., derived from groups by the gateway.
    """

    model_config = ConfigDict(extra="ignore")

    tenant_id: str = "default"
    principal: Optional[str] = None
    agent_id: Optional[str] = None
    user_tenant_id: Optional[str] = None
    groups: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)

    @classmethod
    def from_user_id(cls, user_id: str) -> "SecurityContext":
        user_id = str(user_id).strip()
        if not user_id:
            raise ValueError("user_id cannot be empty")
        return cls(principal=f"user:{user_id}")

    @field_validator("tenant_id", "principal", "agent_id", "user_tenant_id", mode="before")
    @classmethod
    def _validate_optional_non_empty_strings(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} cannot be empty")
        return stripped

    @field_validator("groups", "roles", mode="before")
    @classmethod
    def _validate_string_lists(cls, v: Any, info: ValidationInfo) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"{info.field_name} must be a list of strings")
        items: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"{info.field_name} must be a list of strings")
            stripped = item.strip()
            if stripped:
                items.append(stripped)
        return items
