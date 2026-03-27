import pdfplumber

with pdfplumber.open(r"e:\RTFD_Model\rainfall_pdf\Water_level_&_Rainfall_2026__1774413692.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"--- TEXT PAGE {i} ---")
        text = page.extract_text()
        if text:
            lines = text.split('\n')
            for line in lines[:15]:
                print(line)
        tables = page.extract_tables()
        print(f"--- TABLES PAGE {i} ---")
        for t in tables:
            for r in t[:5]:
                print(r)
            for r in t:
                if "Baddegama" in str(r) or "Thawalama" in str(r) or "baddegama" in str(r) or "thawalama" in str(r):
                    print("FOUND MENTION: ", r)
