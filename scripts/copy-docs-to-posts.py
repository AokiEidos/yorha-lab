#!/usr/bin/env python3
"""Copy docs md files to yorha posts with flattened names and frontmatter."""

import os
import re
from datetime import datetime

SRC_ROOT = "/Users/haru/workspace/docs"
DST_ROOT = "/Users/haru/workspace/yorha-lab/content/posts"

SERIES_MAP = {
    "diffusion": {"tag": "diffusion"},
    "vla": {"tag": "VLA"},
}

def extract_h1(content):
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else "未命名"

def gen_frontmatter(title, tag, date_str):
    return f'''---
title: "{title}"
date: {date_str}
draft: false
tags: ["{tag}"]
hidden: true
---

'''

def process_file(src_path, dst_fname, tag):
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    title = extract_h1(content)
    mtime = os.path.getmtime(src_path)
    date = datetime.fromtimestamp(mtime)
    date_str = date.strftime('%Y-%m-%dT%H:%M:%S.000+08:00')

    new_content = gen_frontmatter(title, tag, date_str) + content
    dst_path = os.path.join(DST_ROOT, dst_fname)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  ✓ {dst_fname}")

def main():
    for series_key, info in SERIES_MAP.items():
        src_dir = os.path.join(SRC_ROOT, series_key)
        print(f"\n处理 {series_key} 系列:")

        for root, dirs, files in os.walk(src_dir):
            dirs.sort()  # 保证目录顺序
            files.sort()  # 保证文件顺序
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue
                src_path = os.path.join(root, fname)

                # 生成扁平化文件名：{series}-{子目录名}-{文件名}
                rel_dir = os.path.relpath(root, src_dir)
                if rel_dir == ".":
                    # 根目录的文件（如 index.md、glossary.md）
                    dst_fname = f"{series_key}-{fname}"
                else:
                    # 子目录的文件
                    subdir = rel_dir.replace("/", "-")
                    dst_fname = f"{series_key}-{subdir}-{fname}"

                process_file(src_path, dst_fname, info["tag"])

    print("\n完成！")

if __name__ == "__main__":
    main()
