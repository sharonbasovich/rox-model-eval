"""Stateful CRM backend for multi-turn agent sessions.

A task's `inputs.crm` holds tables (`accounts`, `contacts`, `deals`, `tasks`). Reads see
earlier writes, so an agent that forgets what it changed two turns ago gets caught by the
end-state checks. Invalid ids and cross-account references return errors, as a real API
would. Empty-string or null arguments count as omitted, since strict structured-output
modes fill every optional field.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .types import ToolCall

PRIMARY_KEYS = {
    "accounts": "account_id",
    "contacts": "contact_id",
    "deals": "deal_id",
    "tasks": "task_id",
    "sent_emails": "email_id",
}
_UPDATABLE = {
    "update_account": ("accounts", {"owner", "tier"}),
    "update_contact": ("contacts", {"title", "email", "status"}),
    "update_deal": ("deals", {"stage", "amount", "close_date", "champion_id"}),
    "update_task": ("tasks", {"title", "due_date", "status"}),
}


class CrmBackend:
    def __init__(self, crm: dict[str, Any]) -> None:
        self.state: dict[str, list[dict[str, Any]]] = copy.deepcopy(crm)
        for table in PRIMARY_KEYS:
            self.state.setdefault(table, [])

    def _find(self, table: str, record_id: Any) -> dict[str, Any] | None:
        key = PRIMARY_KEYS[table]
        return next((r for r in self.state[table] if str(r[key]) == str(record_id)), None)

    def _where(self, table: str, **match: Any) -> list[dict[str, Any]]:
        return [r for r in self.state[table] if all(r.get(k) == v for k, v in match.items())]

    def call(self, call: ToolCall) -> str:
        try:
            args = {k: v for k, v in call.arguments.items() if v not in ("", None)}
            return json.dumps(self._dispatch(call.name, args))
        except LookupError as exc:
            return json.dumps({"error": str(exc)})

    def _require(self, table: str, record_id: Any) -> dict[str, Any]:
        rec = self._find(table, record_id)
        if rec is None:
            raise LookupError(f"no {table[:-1]} with id {record_id!r}")
        return rec

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "search_accounts":
            q = str(args.get("query", "")).strip().lower()
            return [
                {k: a[k] for k in ("account_id", "name", "domain")}
                for a in self.state["accounts"]
                if q and (q in a["name"].lower() or q in a["domain"].lower())
            ]
        if name == "list_accounts":
            owner = args.get("owner")
            rows = self.state["accounts"]
            return [a for a in rows if owner is None or a.get("owner") == owner]
        if name == "get_account":
            return self._require("accounts", args.get("account_id"))
        if name in ("list_contacts", "list_deals", "list_tasks"):
            acc = self._require("accounts", args.get("account_id"))
            return self._where(name.removeprefix("list_"), account_id=acc["account_id"])
        if name == "create_task":
            acc = self._require("accounts", args.get("account_id"))
            for ref, table in (("deal_id", "deals"), ("contact_id", "contacts")):
                if args.get(ref) is not None:
                    rec = self._require(table, args[ref])
                    if rec["account_id"] != acc["account_id"]:
                        raise LookupError(f"{ref} {args[ref]!r} belongs to another account")
            task = {"task_id": f"t_{len(self.state['tasks']) + 1:03d}", "status": "open", **args}
            self.state["tasks"].append(task)
            return task
        if name in _UPDATABLE:
            table, fields = _UPDATABLE[name]
            key = PRIMARY_KEYS[table]
            rec = self._require(table, args.pop(key, None))
            unknown = set(args) - fields
            if unknown:
                raise LookupError(f"cannot update {sorted(unknown)}")
            rec.update(args)
            return rec
        if name == "send_email":
            email = {"email_id": f"e_{len(self.state['sent_emails']) + 1:03d}", **args}
            self.state["sent_emails"].append(email)
            return {"status": "sent", **email}
        raise LookupError(f"tool '{name}' unavailable")
