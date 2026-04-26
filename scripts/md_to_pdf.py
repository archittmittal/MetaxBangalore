import os
from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'ConflictEnv: A Social Negotiation Benchmark', new_x="LMARGIN", new_y="NEXT", align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_pdf(input_md, output_pdf):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    with open(input_md, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    in_code_block = False
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Skip empty lines to avoid ln errors
        if not line.strip() and not in_code_block:
            pdf.ln(5)
            continue

        try:
            if in_code_block:
                pdf.set_font('Courier', '', 8)
                # Truncate extremely long lines that break fpdf
                if len(line) > 100: line = line[:100] + "..."
                pdf.multi_cell(0, 5, line)
            elif line.startswith('# '):
                pdf.set_font('Helvetica', 'B', 18)
                pdf.cell(0, 15, line[2:], new_x="LMARGIN", new_y="NEXT", align='L')
            elif line.startswith('## '):
                pdf.set_font('Helvetica', 'B', 14)
                pdf.cell(0, 10, line[3:], new_x="LMARGIN", new_y="NEXT", align='L')
            elif line.startswith('### '):
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 10, line[4:], new_x="LMARGIN", new_y="NEXT", align='L')
            elif line.startswith('- ') or line.startswith('* '):
                pdf.set_font('Helvetica', '', 11)
                text = f"  • {line[2:]}"
                if len(text) > 500: text = text[:500] + "..."
                pdf.multi_cell(0, 8, text)
            else:
                line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                line = re.sub(r'_(.*?)_', r'\1', line)
                line = line.replace('$', '')
                pdf.set_font('Helvetica', '', 11)
                if len(line) > 1000: line = line[:1000] + "..."
                pdf.multi_cell(0, 8, line)
        except Exception as e:
            print(f"Skipping problematic line: {e}")
            continue

    pdf.output(output_pdf)
    print(f"PDF generated: {output_pdf}")

if __name__ == "__main__":
    create_pdf('docs/ConflictEnv_Paper_Full.md', 'docs/ConflictEnv_Research_Paper.pdf')
