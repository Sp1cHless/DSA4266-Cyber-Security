"""Find CJK text in project-owned text files, filenames and Office document XML."""
import argparse
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SKIP = {".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", "venv", "env"}
TEXT = {".md", ".py", ".ps1", ".sh", ".bat", ".txt", ".json", ".jsonl", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".sql", ".csv", ".tsv", ".ipynb", ".js", ".ts", ".html", ".css", ".xml"}
OFFICE = {".docx", ".xlsx", ".pptx"}


def audit(root):
    findings, checked, office, font_identifiers = [], 0, 0, []
    for parent, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in SKIP and not d.startswith(".venv")]
        for name in files:
            path = Path(parent)/name
            relative = path.relative_to(root).as_posix()
            if PATTERN.search(name):
                findings.append(dict(path=relative, location="filename"))
            if path.suffix.lower() in TEXT:
                checked += 1
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    while chunk := handle.read(1024*1024):
                        if not chunk.isascii() and PATTERN.search(chunk):
                            findings.append(dict(path=relative, location="text"))
                            break
            elif path.suffix.lower() in OFFICE and zipfile.is_zipfile(path):
                office += 1
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if member.endswith(".xml") and member.startswith(("word/", "xl/", "ppt/")):
                            payload = archive.read(member)
                            if payload.isascii():
                                continue
                            document = ET.fromstring(payload)
                            parents = {child: node for node in document.iter() for child in node}
                            for node in document.iter():
                                tag = node.tag.rsplit("}", 1)[-1]
                                xml_parent = parents.get(node)
                                parent_tag = xml_parent.tag.rsplit("}", 1)[-1] if xml_parent is not None else ""
                                if PATTERN.search((node.text or "")+(node.tail or "")):
                                    findings.append(dict(path=relative, location=member, element=tag))
                                for key, value in node.attrib.items():
                                    if not PATTERN.search(value):
                                        continue
                                    attribute = key.rsplit("}", 1)[-1]
                                    is_font = attribute == "typeface" or (tag == "font" and attribute == "name") or (tag == "altName" and attribute == "val") or (tag == "name" and parent_tag == "font" and attribute == "val")
                                    is_builtin_style = tag == "cellStyle" and attribute == "name" and "builtinId" in node.attrib
                                    record = dict(path=relative, location=member, element=tag, attribute=attribute)
                                    (font_identifiers if is_font or is_builtin_style else findings).append(record)
    return dict(passed=not findings, text_files_checked=checked, office_files_checked=office,
                excluded="Git metadata, dependency environments, package directories and bytecode caches",
                scope="CJK detection in project-owned text and Office XML, not proof that every word is English",
                preserved_font_and_builtin_style_identifiers=font_identifiers,
                findings=findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
