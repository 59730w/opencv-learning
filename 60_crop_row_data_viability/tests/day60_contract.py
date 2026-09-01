from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
required = [
    ROOT / "60_crop_row_data_viability" / "code" / "day60_data_audit.py",
    ROOT / "60_crop_row_data_viability" / "code" / "day60_notes.md",
    ROOT / "60_crop_row_data_viability" / "assets" / "README.md",
    ROOT / "projects" / "04_crop_row_perception" / "data_viability_report.md",
    ROOT / "projects" / "04_crop_row_perception" / "data" / "source_registry.json",
    ROOT / "projects" / "04_crop_row_perception" / "data" / "audit_result.json",
]

missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"Missing Day60 files: {missing}")

notes = required[1].read_text(encoding="utf-8")
report = required[3].read_text(encoding="utf-8")
project_readme = (ROOT / "projects" / "04_crop_row_perception" / "README.md").read_text(
    encoding="utf-8"
)

for text, name in ((notes, "notes"), (report, "report"), (project_readme, "README")):
    if "BLOCKED" not in text:
        raise SystemExit(f"Day60 {name} must preserve the blocked gate")
    if "DAY60_DATA_VIABILITY_PASS" in text:
        raise SystemExit(f"Day60 {name} contains a false PASS claim")

print("DAY60_CONTRACT_OK")
