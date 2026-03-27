import json
from pathlib import Path
from pdf_watcher import parse_pdf

pdfs = [
    Path(r"e:\RTFD_Model\rainfall_pdf\Water_level_&_Rainfall_2026__1774498675.pdf"),
    Path(r"e:\RTFD_Model\rainfall_pdf\Water_level_&_Rainfall_2026__1774413692.pdf")
]

for p in pdfs:
    print(f"--- parsing {p.name} ---")
    data = parse_pdf(p)
    print(json.dumps(data, indent=2))
