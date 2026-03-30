import fitz
doc = fitz.open('Essentials.pdf')
# Read key sections: SA (p23-26), DE (p50-52), PSO (p52-54), Hybrid (p47-49)
for page_num in [22, 23, 24, 25, 49, 50, 51, 52, 53, 46, 47]:
    text = doc[page_num].get_text()
    if text.strip():
        print(f'\n=== PAGE {page_num+1} ===')
        print(text[:4000])
