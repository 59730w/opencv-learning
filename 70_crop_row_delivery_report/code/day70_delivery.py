"""Verify and summarize the evidence-bounded Practical 04 delivery."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SUCCESS_MARKER = "DAY70_DELIVERY_CHECK_COMPLETE"
REQUIRED_CLAIM_STATUSES = {
    "offline_pilot_delivery": "PASS",
    "ssr_central_row_recall": "FAILED",
    "cross_device_exact_parity": "FAILED",
    "all_row_external_generalization": "NOT_ESTABLISHED",
    "reject_aware_external_generalization": "BLOCKED",
    "real_video_safety": "BLOCKED",
    "metric_robot_measurement": "BLOCKED",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {relative_path}")
    return candidate


def _json_value(payload: Any, dotted_path: str) -> Any:
    value = payload
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, **details: Any) -> None:
    checks.append({"id": check_id, "passed": bool(passed), **details})


def _candidate_repository_paths(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def verify_delivery(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    checks: list[dict[str, Any]] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _check(
        checks,
        "manifest:identity",
        manifest.get("schema_version") == 1
        and manifest.get("marker") == "DAY70_DELIVERY_MANIFEST"
        and manifest.get("date") == "2026-09-11",
        actual={key: manifest.get(key) for key in ("schema_version", "marker", "date")},
    )

    for item in manifest.get("frozen_artifacts", []):
        relative_path = item["path"]
        path = _resolve_inside(repo_root, relative_path)
        actual = sha256_file(path) if path.is_file() else None
        _check(
            checks,
            f"frozen_hash:{relative_path}",
            actual == item["sha256"],
            expected=item["sha256"],
            actual=actual,
            role=item.get("role"),
        )

    evidence: dict[str, Any] = {}
    loaded_sources: dict[Path, Any] = {}
    for item in manifest.get("evidence_checks", []):
        source = _resolve_inside(repo_root, item["source"])
        try:
            if source not in loaded_sources:
                loaded_sources[source] = json.loads(source.read_text(encoding="utf-8"))
            actual = _json_value(loaded_sources[source], item["json_path"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            actual = None
        expected = item["expected"]
        evidence[item["id"]] = actual
        _check(
            checks,
            f"evidence:{item['id']}",
            actual == expected,
            source=item["source"],
            json_path=item["json_path"],
            expected=expected,
            actual=actual,
        )

    claims = manifest.get("claims", [])
    claims_by_id = {item.get("id"): item for item in claims}
    for claim_id, expected_status in REQUIRED_CLAIM_STATUSES.items():
        claim = claims_by_id.get(claim_id, {})
        _check(
            checks,
            f"claim:{claim_id}",
            claim.get("status") == expected_status and bool(claim.get("statement")),
            expected=expected_status,
            actual=claim.get("status"),
        )
    unexpected_claims = sorted(set(claims_by_id) - set(REQUIRED_CLAIM_STATUSES))
    _check(checks, "claim:unexpected", not unexpected_claims, actual=unexpected_claims)

    runtime = manifest.get("runtime_contract", {})
    if runtime:
        launcher = _resolve_inside(repo_root, runtime["launcher"])
        launcher_is_ascii = launcher.is_file() and launcher.read_bytes().isascii()
        _check(
            checks,
            "runtime:launcher_ascii",
            not runtime.get("launcher_ascii") or launcher_is_ascii,
            actual=launcher_is_ascii,
        )
        ui_source = _resolve_inside(repo_root, runtime["ui_source"])
        ui_text = ui_source.read_text(encoding="utf-8") if ui_source.is_file() else ""
        missing_tokens = [token for token in runtime.get("required_ui_tokens", []) if token not in ui_text]
        _check(checks, "runtime:ui_tokens", not missing_tokens, actual=missing_tokens)

    for relative_path in manifest.get("required_artifacts", []):
        exists = _resolve_inside(repo_root, relative_path).is_file()
        _check(checks, f"required:{relative_path}", exists, actual=exists)

    for contract in manifest.get("text_contracts", []):
        relative_path = contract["path"]
        text_path = _resolve_inside(repo_root, relative_path)
        text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
        missing = [token for token in contract.get("required_tokens", []) if token not in text]
        forbidden_present = [token for token in contract.get("forbidden_tokens", []) if token in text]
        _check(
            checks,
            f"text:{relative_path}",
            not missing and not forbidden_present,
            missing=missing,
            forbidden_present=forbidden_present,
        )

    hygiene = manifest.get("tracked_hygiene", {})
    if hygiene.get("enabled"):
        try:
            paths = _candidate_repository_paths(repo_root)
            forbidden = [
                path
                for path in paths
                if any(re.search(pattern, path, flags=re.IGNORECASE) for pattern in hygiene["forbidden_patterns"])
            ]
            _check(checks, "repository:tracked_hygiene", not forbidden, actual=forbidden)
        except (OSError, subprocess.CalledProcessError) as error:
            _check(checks, "repository:tracked_hygiene", False, actual=str(error))

    failures = [item for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "marker": SUCCESS_MARKER,
        "date": manifest.get("date"),
        "baseline_commit": manifest.get("baseline_commit"),
        "passed": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
        "failures": failures,
        "evidence": evidence,
        "claims": claims,
    }


def render_evidence_board(report: dict[str, Any], output_path: Path) -> Path:
    colors = {
        "PASS": ("#B8F2D0", "#0E4A35"),
        "FAILED": ("#FFD0C7", "#70281D"),
        "BLOCKED": ("#FFE6A6", "#654900"),
        "NOT_ESTABLISHED": ("#DCE3EA", "#35424D"),
    }
    claims = report.get("claims", [])
    evidence = report.get("evidence", {})

    def compact_number(key: str, digits: int | None = None) -> str:
        value = evidence.get(key)
        if value is None:
            return "n/a"
        if digits is None:
            return str(value)
        return f"{float(value):.{digits}f}"

    metric_line = (
        f"CROW {compact_number('crow_frozen_frames')} frames · nav leakage "
        f"{compact_number('crow_navigation_contract_violations')}   |   SSR matched "
        f"{compact_number('ssr_matched_references')}/{compact_number('ssr_images')} · pos MAE "
        f"{compact_number('ssr_position_mae_norm', 4)} · heading MAE "
        f"{compact_number('ssr_heading_mae_deg', 3)} deg"
    )
    cards = []
    for index, claim in enumerate(claims):
        column = index % 2
        row = index // 2
        is_last_odd_card = index == len(claims) - 1 and len(claims) % 2 == 1
        x = 90 + column * 810
        y = 350 + row * 145
        width = 1550 if is_last_odd_card else 740
        status = str(claim["status"])
        fill, ink = colors.get(status, ("#E7E7E7", "#242424"))
        label = status.replace("_", " ")
        statement = html.escape(str(claim["statement"]))
        cards.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="128" rx="22" fill="{fill}"/>'
            f'<text x="{x + 32}" y="{y + 43}" class="status" fill="{ink}">{label}</text>'
            f'<text x="{x + 32}" y="{y + 91}" class="body" fill="#17251F">{statement}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1080" viewBox="0 0 1800 1080">
<rect width="1800" height="1080" fill="#F4F2E8"/>
<rect x="0" y="0" width="1800" height="275" fill="#12372F"/>
<path d="M1120 275 L1800 25 M1320 275 L1800 100 M1520 275 L1800 180" stroke="#F2C85B" stroke-width="5" opacity=".45"/>
<text x="90" y="94" class="kicker">PRACTICAL 04 · DELIVERY EVIDENCE</text>
<text x="90" y="175" class="title">Offline pilot, bounded claims</text>
<text x="90" y="225" class="subtitle">Frozen evidence · Only valid may expose a corridor center</text>
<text x="90" y="315" class="metrics">{html.escape(metric_line)}</text>
{''.join(cards)}
<text x="90" y="1040" class="footer">External central-row recall 0.7143 &lt; 0.80 · Real-video safety remains BLOCKED</text>
<style>
.title{{font:700 54px 'Segoe UI',Arial,sans-serif;fill:#F8F4E8}}
.subtitle{{font:24px 'Segoe UI',Arial,sans-serif;fill:#CEDDD5}}
.kicker{{font:600 18px Consolas,monospace;letter-spacing:4px;fill:#F2C85B}}
.metrics{{font:19px Consolas,monospace;fill:#55645D}}
.status{{font:700 25px Consolas,monospace;letter-spacing:2px}}
.body{{font:22px 'Microsoft YaHei UI','Segoe UI',Arial,sans-serif}}
.footer{{font:22px Consolas,monospace;fill:#55645D}}
</style></svg>'''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(svg)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = verify_delivery(args.repo_root, args.manifest)
    render_evidence_board(report, args.board)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(SUCCESS_MARKER if report["passed"] else "DAY70_DELIVERY_CHECK_FAILED")
    print(f"checks={report['check_count']} failures={report['failure_count']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
