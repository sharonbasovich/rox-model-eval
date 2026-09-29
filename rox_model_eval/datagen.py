"""Deterministic generators for bulk data-operation tasks (C10).

Each generator takes `{kind, seed, n}` and returns `(data_text, expected)`. The expected
answer is computed from the generated rows by applying the task's stated rules, never from
how the rows were constructed, so a generator bug cannot silently make the key disagree
with the instructions.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

_FIRST = [
    "Dana",
    "Leo",
    "Mina",
    "Ravi",
    "Aiko",
    "Omar",
    "Lucia",
    "Ben",
    "Priya",
    "Sam",
    "Jordan",
    "Elena",
    "Marco",
    "Tessa",
    "Victor",
    "Nadia",
    "Hugo",
    "Ines",
    "Kofi",
    "Maya",
    "Felix",
    "Rosa",
    "Theo",
    "Zara",
    "Ivan",
    "Lena",
    "Arjun",
    "Chloe",
    "Diego",
    "Hana",
]
_LAST = [
    "Okafor",
    "Brandt",
    "Sato",
    "Menon",
    "Tanaka",
    "Faris",
    "Gomez",
    "Ode",
    "Nair",
    "Ortiz",
    "Blake",
    "Vasquez",
    "Bianchi",
    "Lindqvist",
    "Moreau",
    "Haddad",
    "Keller",
    "Duarte",
    "Mensah",
    "Kaplan",
    "Novak",
    "Reyes",
    "Fischer",
    "Ali",
    "Petrov",
    "Weber",
    "Rao",
]
_COMPANIES = [
    ("Harbor Health", "harborhealth.com"),
    ("Harbor Health Partners", "hhpartners.com"),
    ("Tidepool", "tidepool.io"),
    ("Ferrovane", "ferrovane.com"),
    ("Quillmesh", "quillmesh.ai"),
    ("Brightkeel Logistics", "brightkeel.com"),
    ("Kestrel Logistics", "kestrel-log.com"),
    ("Tessaract Freight", "tessaract.io"),
    ("Saltmarsh Coffee", "saltmarshcoffee.co"),
    ("Northwind Analytics", "northwind-analytics.com"),
    ("Lumen Grid", "lumengrid.com"),
    ("Orchard Bio", "orchardbio.com"),
]


def _phone(rng: random.Random) -> str:
    return f"415{rng.randint(200, 999)}{rng.randint(1000, 9999)}"


def _fmt_phone(digits: str, rng: random.Random) -> str:
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"+1 {a} {b} {c}", f"{a}.{b}.{c}"])


def dedupe_contacts(rng: random.Random, n: int) -> tuple[str, dict[str, Any]]:
    people: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    while len(people) < n * 3 // 4:
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        company, domain = rng.choice(_COMPANIES)
        if (first + last, company) in seen:
            continue
        seen.add((first + last, company))
        people.append(
            {
                "name": f"{first} {last}",
                "email": f"{first[0].lower()}{last.lower()}@{domain}",
                "company": company,
                "phone": _phone(rng),
            }
        )
    rows = [dict(p) for p in people]
    while len(rows) < n:
        src = rng.choice(people)
        kind = rng.random()
        dup = dict(src)
        if kind < 0.4:
            dup["email"] = (
                f"  {src['email'].upper()} " if rng.random() < 0.5 else src["email"].title()
            )
            dup["phone"] = _phone(rng)
        elif kind < 0.7:
            first = src["name"].split()[0].lower()
            dup["email"] = f"{first}.{rng.randint(10, 99)}@gmail.com"
            dup["company"] = src["company"].upper() if rng.random() < 0.5 else src["company"]
        elif kind < 0.85:
            other_company = rng.choice([c for c, _ in _COMPANIES if c != src["company"]])
            dup["company"] = other_company
            dup["email"] = f"{src['name'].split()[0].lower()}@{rng.choice(_COMPANIES)[1]}"
        else:
            dup["email"] = (
                f"{src['name'].split()[0].lower()}.{src['name'].split()[1].lower()}@{rng.choice(_COMPANIES)[1]}"
            )
            dup["phone"] = _phone(rng)
        rows.append(dup)
    rng.shuffle(rows)
    ids = [f"r{i + 1:03d}" for i in range(len(rows))]

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def key_email(r: dict[str, str]) -> str:
        return r["email"].strip().lower()

    def key_phone(r: dict[str, str]) -> tuple[str, str]:
        return (r["company"].strip().lower(), re.sub(r"\D", "", r["phone"])[-10:])

    for key in (key_email, key_phone):
        first_seen: dict[Any, int] = {}
        for i, r in enumerate(rows):
            k = key(r)
            if k in first_seen:
                parent[find(i)] = find(first_seen[k])
            else:
                first_seen[k] = i
    clusters: dict[int, list[str]] = {}
    for i in range(len(rows)):
        clusters.setdefault(find(i), []).append(ids[i])
    groups = sorted(sorted(g) for g in clusters.values() if len(g) > 1)

    lines = ["id,name,email,company,phone"]
    lines += [
        f'{ids[i]},{r["name"]},"{r["email"]}",{r["company"]},{_fmt_phone(r["phone"], rng)}'
        for i, r in enumerate(rows)
    ]
    return "\n".join(lines), {"groups": {"duplicate_groups": groups}, "valid_ids": ids}


def _norm_domain(d: str) -> str:
    d = d.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    return d.rstrip("/")


def reconcile_billing(rng: random.Random, n: int) -> tuple[str, dict[str, Any]]:
    stems = [
        f"{a}{b}"
        for a in (
            "tide",
            "iron",
            "blue",
            "fern",
            "gold",
            "opal",
            "sage",
            "ember",
            "north",
            "cedar",
            "lark",
            "moss",
            "quartz",
        )
        for b in (
            "labs",
            "works",
            "hq",
            "ly",
            "grid",
            "flow",
            "base",
            "stack",
            "mint",
            "loop",
            "bay",
            "path",
        )
    ]
    rng.shuffle(stems)
    domains = [f"{s}.com" for s in stems[:n]]
    crm_only = set(rng.sample(range(n), n // 15))
    billing_only = set(rng.sample(sorted(set(range(n)) - crm_only), n // 15))
    crm: list[dict[str, Any]] = []
    billing: list[dict[str, Any]] = []
    for i, dom in enumerate(domains):
        arr = rng.randrange(6000, 240000, 600)
        if i not in billing_only:
            crm.append({"id": f"a{i + 1:03d}", "domain": dom, "arr": arr})
        if i not in crm_only:
            roll = rng.random()
            mrr = round(arr / 12, 2)
            if roll < 0.12:
                mrr = round(mrr + rng.choice([-1, 1]) * rng.randrange(50, 2000, 25), 2)
            shown = dom
            style = rng.random()
            if style < 0.2:
                shown = f"https://www.{dom}/"
            elif style < 0.35:
                shown = dom.upper()
            elif style < 0.45:
                shown = f"www.{dom}"
            elif style < 0.5:
                shown = f"app.{dom}"
            billing.append({"id": f"b{len(billing) + 1:03d}", "domain": shown, "mrr": mrr})
    rng.shuffle(billing)
    billing = [dict(b, id=f"b{j + 1:03d}") for j, b in enumerate(billing)]

    by_domain = {c["domain"]: c for c in crm}
    matched: set[str] = set()
    mismatches, missing_in_crm = [], []
    for b in billing:
        c = by_domain.get(_norm_domain(b["domain"]))
        if c is None:
            missing_in_crm.append(b["id"])
            continue
        matched.add(c["id"])
        if abs(c["arr"] - 12 * b["mrr"]) > 1:
            mismatches.append(c["id"])
    missing_in_billing = [c["id"] for c in crm if c["id"] not in matched]

    text = "CRM ACCOUNTS (id,domain,arr_usd)\n" + "\n".join(
        f"{c['id']},{c['domain']},{c['arr']}" for c in crm
    )
    text += "\n\nBILLING EXPORT (billing_id,customer_domain,mrr_usd)\n" + "\n".join(
        f"{b['id']},{b['domain']},{b['mrr']:.2f}" for b in billing
    )
    return text, {
        "sets": {
            "arr_mismatches": sorted(mismatches),
            "missing_in_crm": sorted(missing_in_crm),
            "missing_in_billing": sorted(missing_in_billing),
        },
        "valid_ids": [c["id"] for c in crm] + [b["id"] for b in billing],
    }


_RATES = {"USD": 1.0, "EUR": 1.10, "GBP": 1.25}


def pipeline_rollup(rng: random.Random, n: int) -> tuple[str, dict[str, Any]]:
    owners = [
        "Priya Nair",
        "Priya Naidu",
        "Sam Ortiz",
        "Jordan Blake",
        "Elena Vasquez",
        "Kofi Mensah",
    ]
    stages = ["discovery", "proposal", "negotiation", "closed_won", "closed_lost"]
    months = ["2026-09", "2026-10", "2026-11", "2026-12", "2027-01"]
    lines = ["deal_id,owner,stage,close_date,amount,currency"]
    totals = {o: 0.0 for o in owners}
    for i in range(n):
        owner, stage = rng.choice(owners), rng.choice(stages)
        month = rng.choice(months)
        day = (
            rng.choice([1, 2, 15, 28, 30, 31])
            if month in ("2026-10", "2026-12")
            else rng.randint(1, 28)
        )
        if month == "2026-09" and rng.random() < 0.3:
            day = 30
        close = f"{month}-{day:02d}"
        cur = rng.choice(["USD", "USD", "EUR", "GBP"])
        amount = rng.randrange(5000, 180000, 500)
        lines.append(f"d{i + 1:04d},{owner},{stage},{close},{amount},{cur}")
        if stage in stages[:3] and "2026-10-01" <= close <= "2026-12-31":
            totals[owner] += amount * _RATES[cur]
    return "\n".join(lines), {
        "values": {"q4_open_pipeline_usd": {o: round(v) for o, v in totals.items()}}
    }


def csv_import_validation(rng: random.Random, n: int) -> tuple[str, dict[str, Any]]:
    existing = sorted(
        {
            f"{rng.choice(_FIRST).lower()}.{rng.choice(_LAST).lower()}@{rng.choice(_COMPANIES)[1]}"
            for _ in range(30)
        }
    )
    lines = ["row_id,name,email,company,employee_count"]
    invalid: dict[str, str] = {}
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
    for i in range(n):
        rid = f"row{i + 1:03d}"
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        company, domain = rng.choice(_COMPANIES)
        email = f"{first.lower()}.{last.lower()}{rng.randint(1, 999)}@{domain}"
        emp = str(rng.randint(5, 5000))
        roll = rng.random()
        if roll < 0.05:
            company = rng.choice(["", " "])
        elif roll < 0.10:
            email = rng.choice(
                [
                    email.replace("@", ""),
                    email.replace("@", "@@"),
                    email.replace(".", " ", 1),
                    email.rsplit(".", 1)[0],
                    email.replace("@", " @"),
                ]
            )
        elif roll < 0.15:
            emp = rng.choice(["-12", "12.5", "unknown", "1,200", "~300"])
        elif roll < 0.21:
            e = rng.choice(existing)
            email = rng.choice([e, e.upper(), f" {e}"])
        elif roll < 0.24:
            company, email = "", f"{first.lower()}@@{domain}"
        elif roll < 0.26:
            emp = ""
        name = f"{first} {last}"
        lines.append(f'{rid},{name},"{email}","{company}",{emp}')
        e_clean = email.strip()
        if not company.strip():
            invalid[rid] = "missing_company"
        elif not email_re.match(e_clean):
            invalid[rid] = "invalid_email"
        elif emp and not emp.isdigit():
            invalid[rid] = "bad_employee_count"
        elif e_clean.lower() in existing:
            invalid[rid] = "duplicate_existing"
    text = "EXISTING CRM EMAILS\n" + "\n".join(existing) + "\n\nIMPORT FILE\n" + "\n".join(lines)
    return text, {"values": {"invalid_rows": invalid}}


GENERATORS: dict[str, Callable[[random.Random, int], tuple[str, dict[str, Any]]]] = {
    "dedupe_contacts": dedupe_contacts,
    "reconcile_billing": reconcile_billing,
    "pipeline_rollup": pipeline_rollup,
    "csv_import_validation": csv_import_validation,
}


def build_dataset(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gen = GENERATORS[str(spec["kind"])]
    return gen(random.Random(spec.get("seed", 0)), int(spec.get("n", 200)))


def reference_answer(expected: dict[str, Any]) -> dict[str, Any]:
    """The JSON object a perfect answer would return."""
    out: dict[str, Any] = {}
    for section in ("sets", "groups", "values"):
        out |= expected.get(section, {})
    return out
