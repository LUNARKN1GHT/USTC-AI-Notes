#!/usr/bin/env python3
"""Prepare the public Markdown tree consumed by MkDocs."""

from pathlib import Path
import os
import re
import shutil


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / ".build" / "docs"
PUBLIC_DIRECTORIES = (
    "assets",
    "00-公共基础",
    "00-数学基础",
    "01-人工智能概述",
    "02-机器学习基础",
    "03-神经网络基础",
    "04-图神经网络",
    "05-Transformer",
    "06-自监督学习",
    "07-强化学习",
)
PUBLIC_FILES = (
    "人工智能数学原理与算法A.md",
    "CONTRIBUTING.md",
    "ATTRIBUTION.md",
    "LICENSE",
)

WIKILINK = re.compile(r"(!?)\[\[([^\]]+)\]\]")
INLINE_CODE = re.compile(r"(`+[^`]*?`+)")


def aliases_for(path: Path) -> list[str]:
    """Read the simple YAML alias list used by the notes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return []

    aliases: list[str] = []
    in_aliases = False
    for line in lines[1:]:
        if line == "---":
            break
        if line == "aliases:":
            in_aliases = True
            continue
        if in_aliases:
            match = re.match(r"\s+-\s+(.+?)\s*$", line)
            if match:
                aliases.append(match.group(1).strip('"\''))
            elif line and not line[0].isspace():
                in_aliases = False
    return aliases


def build_link_index() -> tuple[dict[str, Path], dict[str, Path]]:
    pages: dict[str, Path] = {}
    assets: dict[str, Path] = {}
    paths = sorted(path for path in DOCS_DIR.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(DOCS_DIR)
        assets.setdefault(path.name, relative)
        if path.suffix.lower() == ".md":
            pages[path.stem] = relative
    for path in paths:
        if path.suffix.lower() != ".md":
            continue
        relative = path.relative_to(DOCS_DIR)
        for alias in aliases_for(path):
            pages.setdefault(alias, relative)
    return pages, assets


def convert_wikilinks(path: Path, pages: dict[str, Path], assets: dict[str, Path]) -> None:
    """Convert Obsidian links in the staged copy without touching source notes."""
    source = path.read_text(encoding="utf-8")
    in_fence = False
    output: list[str] = []

    def replace(match: re.Match[str]) -> str:
        embedded, body = match.groups()
        parts = body.replace(r"\|", "|").split("|")
        target = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else target
        page_name, separator, heading = target.partition("#")

        if not page_name:
            href = f"#{heading}"
        else:
            direct = Path(page_name)
            if direct.suffix.lower() != ".md":
                direct = direct.with_suffix(".md")
            direct_path = DOCS_DIR / direct
            resolved = direct if direct_path.is_file() else pages.get(Path(page_name).name)
            if embedded:
                resolved = assets.get(Path(page_name).name)
            if resolved is None:
                return label
            href = Path(os.path.relpath(DOCS_DIR / resolved, path.parent)).as_posix()
            if separator:
                href += f"#{heading}"

        alt = label.split("|", 1)[0]
        return f"![{alt}](<{href}>)" if embedded else f"[{label}](<{href}>)"

    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue
        chunks = INLINE_CODE.split(line)
        output.append("".join(chunk if chunk.startswith("`") else WIKILINK.sub(replace, chunk) for chunk in chunks))

    path.write_text("".join(output), encoding="utf-8")


def main() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True)

    shutil.copy2(ROOT / "README.md", DOCS_DIR / "index.md")
    for filename in PUBLIC_FILES:
        shutil.copy2(ROOT / filename, DOCS_DIR / filename)
    for dirname in PUBLIC_DIRECTORIES:
        shutil.copytree(ROOT / dirname, DOCS_DIR / dirname)

    pages, assets = build_link_index()
    for path in DOCS_DIR.rglob("*.md"):
        convert_wikilinks(path, pages, assets)

    print(f"Prepared public notes in {DOCS_DIR}")


if __name__ == "__main__":
    main()
