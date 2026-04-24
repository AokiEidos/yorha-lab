#!/usr/bin/env python3
"""Fix relative links in appendix index files after flattening."""

import os
import re

BASE = "/Users/haru/workspace/yorha-lab/content/posts"

# 旧路径片段 → 新前缀
REPLACEMENTS = {
    # Diffusion
    "../01-fundamentals/": "../diffusion-01-fundamentals-",
    "../02-models-zoo/": "../diffusion-02-models-zoo-",
    "../03-conditional-generation/": "../diffusion-03-conditional-generation-",
    "../04-acceleration-deployment/": "../diffusion-04-acceleration-deployment-",
    "../05-llm-era/": "../diffusion-05-llm-era-",
    "../06-autonomous-driving/": "../diffusion-06-autonomous-driving-",
    # VLA
    "../01-foundations/": "../vla-01-foundations-",
    "../02-architecture/": "../vla-02-architecture-",
    "../03-models-zoo/": "../vla-03-models-zoo-",
    "../04-training/": "../vla-04-training-",
    "../05-robotics/": "../vla-05-robotics-",
    "../06-autonomous-driving/": "../vla-06-autonomous-driving-",
    "../07-foundation-models/": "../vla-07-foundation-models-",
}

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original = content
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ {os.path.relpath(filepath, BASE)}")
    else:
        print(f"  — {os.path.relpath(filepath, BASE)} (no change)")

def main():
    files = [
        "diffusion/diffusion-07-appendix-01-index.md",
        "diffusion/diffusion-05-llm-era-07-multimodal.md",
        "vla/vla-08-appendix-01-index.md",
    ]
    for f in files:
        fp = os.path.join(BASE, f)
        if os.path.exists(fp):
            process_file(fp)
        else:
            print(f"  ✗ missing: {f}")
    print("Done")

if __name__ == "__main__":
    main()
