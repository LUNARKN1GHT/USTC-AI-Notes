#!/usr/bin/env bash
# 从私有知识库把「人工智能数学原理与算法」课程笔记同步到本公开仓库。
# 自动排除:讲义/作业 PDF、考试重点、aider 对话记录、.DS_Store、Obsidian 配置。
# 用法:在本仓库目录下执行  ./sync-from-private.sh   然后自行 git add/commit/push。
set -euo pipefail

SRC="/Users/liyifan/Knowledge-Base/99-课程笔记/人工智能数学原理与算法A/"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

rsync -av --delete \
  --exclude='.git/' \
  --exclude='.gitignore' \
  --exclude='README.md' \
  --exclude='LICENSE' \
  --exclude='sync-from-private.sh' \
  --exclude='.obsidian/' \
  --exclude='.aider*' \
  --exclude='.DS_Store' \
  --exclude='*.pdf' \
  --exclude='课后作业/' \
  --exclude='26春期末重点.png' \
  "$SRC" "$DST"

echo
echo "同步完成。请检查改动后执行:"
echo "  git -C \"$DST\" add -A && git -C \"$DST\" commit -m '更新笔记' && git -C \"$DST\" push"
