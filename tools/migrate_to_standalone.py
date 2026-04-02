#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mmdet3d"
DST_ROOT = ROOT / "RCDFNet-main"
DST = DST_ROOT / "rcdfnet"

COPY_DIRS = [
    "apis",
    "core",
    "datasets",
    "models",
    "ops",
    "utils",
]
COPY_FILES = [
    "__init__.py",
    "version.py",
]

EXTRA_TOOLS = [
    (ROOT / "tools" / "train3.py", DST_ROOT / "tools" / "train.py"),
    (ROOT / "tools" / "result_test.py", DST_ROOT / "tools" / "test.py"),
]


def _rewrite_py(file_path: Path) -> None:
    text = file_path.read_text(encoding="utf-8")

    # mmdet3d -> rcdfnet
    text = re.sub(r"\bfrom\s+mmdet3d(\b|\.)", "from rcdfnet\\1", text)
    text = re.sub(r"\bimport\s+mmdet3d(\b|\.)", "import rcdfnet\\1", text)

    # 兼容直接 `import mmdet3d as xxx` 的情况
    text = text.replace("import rcdfnet as", "import rcdfnet as")

    # 避免工具脚本里出现 hardcode 根路径的 PYTHONPATH 提示（仅保持中性）
    text = text.replace("MMDet", "RCDFNet-Standalone")

    file_path.write_text(text, encoding="utf-8")


def _copy_tree(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    for d in COPY_DIRS:
        _copy_tree(SRC / d, DST / d)

    for f in COPY_FILES:
        shutil.copy2(SRC / f, DST / f)

    # 拷贝工具脚本
    for src_tool, dst_tool in EXTRA_TOOLS:
        dst_tool.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tool, dst_tool)

    # 批量改写导入
    for py in list(DST.rglob("*.py")) + [p[1] for p in EXTRA_TOOLS if p[1].suffix == ".py"]:
        _rewrite_py(py)

    print("[OK] RCDFNet-main/rcdfnet migration completed")


if __name__ == "__main__":
    main()
