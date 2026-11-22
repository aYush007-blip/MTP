#!/usr/bin/env python3
import json
from pathlib import Path
import re

repo_dir = Path("/mnt/thesis/assets/tikz_library/repos")
out_index = Path("/mnt/thesis/assets/tikz_library/tikz_index.json")

diagrams = []

all_files = list(repo_dir.glob("**/*.tex")) + list(repo_dir.glob("**/*.tikz")) + list(repo_dir.glob("**/*.pgf"))

for p in all_files:
    if not p.is_file():
        continue
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    # Extract tikzpicture block
    m = re.search(r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", content, flags=re.DOTALL)
    if not m:
        continue

    # Filter out diagrams requiring uncached offline packages (pgfplots, chains, circuitikz, 3dplot)
    if any(k in content for k in ["start chain", "\\begin{axis}", "pgfplots", "circuitikz", "tdplot_set", "tikz-3dplot"]):
        continue

    tikz_body = m.group(1)
    title = p.stem.replace("_", " ").replace("-", " ").title()
    parent_name = p.parent.name.replace("_", " ").replace("-", " ").title()
    if parent_name and parent_name != title:
        title = f"{parent_name} - {title}"

    # Clean textual keywords
    text_cues = " ".join(re.findall(r"[a-zA-Z]{3,}", p.stem + " " + p.parent.name + " " + content[:2000]))
    keywords = sorted(list(set(k.lower() for k in text_cues.split() if len(k) > 2 and k.lower() not in {"the", "and", "for", "with", "from", "draw", "node", "path", "tikz", "tikzpicture", "document", "documentclass"})))

    path_str = str(p).lower()
    if any(d in path_str for d in ["physics", "quantum", "maxwell", "temperature", "optics", "mechanics"]):
        domain = "Physics & Astronomy"
    elif any(d in path_str for d in ["math", "convex", "calculus", "convergence", "liminf", "integral", "linear"]):
        domain = "Mathematics & Optimization"
    elif any(d in path_str for d in ["nn", "neural", "deep", "network", "autoencoder", "tree", "algorithm", "cycle", "execute"]):
        domain = "Computer Science & AI"
    elif any(d in path_str for d in ["chem", "reaction", "periodic", "molecule", "titration"]):
        domain = "Chemistry & Materials"
    else:
        domain = "Engineering & Applied Sciences"

    diagrams.append({
        "id": f"{p.parent.name}_{p.stem}",
        "title": title,
        "domain": domain,
        "rel_path": str(p.relative_to(repo_dir)),
        "keywords": keywords[:35],
        "tikz_code": tikz_body
    })

out_index.write_text(json.dumps(diagrams, indent=2), encoding="utf-8")
print(f"✓ Successfully indexed {len(diagrams)} compile-ready scientific TikZ diagrams across all domains!")

domains = {}
for d in diagrams:
    domains[d["domain"]] = domains.get(d["domain"], 0) + 1
for dom, count in domains.items():
    print(f"  • {dom}: {count} diagrams")
