from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_management.corpus_naming import migrate_corpus_display_names  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="统一测试语料库显示名称")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入")
    args = parser.parse_args()
    changes = migrate_corpus_display_names(args.data_root, dry_run=args.dry_run)
    print(json.dumps(changes, ensure_ascii=False, indent=2))
    print(f"共{'发现' if args.dry_run else '更新'} {len(changes)} 条语料名称")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
