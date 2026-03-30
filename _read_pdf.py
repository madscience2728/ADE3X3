import fitz
doc = fitz.open('Essentials.pdf')
print(f'Pages: {len(doc)}')
toc = doc.get_toc()
for level, title, page in toc:
    indent = "  " * (level - 1)
    print(f'  {indent}{title} (p.{page})')
# Print first 10 pages
for i in range(min(10, len(doc))):
    text = doc[i].get_text()
    if text.strip():
        print(f'\n--- PAGE {i+1} ---')
        print(text[:3000])
