"""
Table and Header-aware PDF loader.

PyPDFLoader extracts raw text and can scramble table layouts or mix key-value headers
(e.g., "Registration No." and student details) with large tables.

This loader extracts structured table rows AND parses colon-separated header lines
(e.g., "Registration No. : 2023BCSE07AED071"), creating focused documents/chunks.
"""

import re
import pdfplumber
from langchain_core.documents import Document


def extract_key_values_and_text(text):
    """
    Extract lines formatted as 'Key : Value' (like Registration No., Student Name)
    into a dedicated header block.
    """
    lines = text.split("\n")
    kv_pairs = []
    normal_lines = []

    kv_regex = re.compile(r"^([A-Za-z0-9\s/&.\-\(\)]+?)\s*:\s*(.+)$")

    for line in lines:
        stripped = line.strip()
        match = kv_regex.match(stripped)
        if match:
            key, val = match.group(1).strip(), match.group(2).strip()
            # Filter out URLs, timestamps, or long table headers
            if not key.lower().startswith(("http", "www")) and len(key) < 45 and not ("Marks" in key or "Min." in key or "Max." in key):
                kv_pairs.append(f"{key}: {val}")
                continue
        normal_lines.append(line)

    return kv_pairs, "\n".join(normal_lines)


def build_table_aware_documents(pdf_path):
    """
    Extract text from a PDF, treating tables and key-value header lines specially.
    """
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            full_text = page.extract_text() or ""
            kv_pairs, remaining_text = extract_key_values_and_text(full_text)

            # If key-value header pairs were found, create a dedicated Document
            if kv_pairs:
                header_block = "\n".join(kv_pairs)
                documents.append(
                    Document(
                        page_content=header_block,
                        metadata={"source": pdf_path, "page": page_num, "section": "header_metadata"},
                    )
                )

            # Extract tables with structure preserved
            tables = page.extract_tables()
            table_text_parts = []

            for table in tables:
                if not table:
                    continue
                for row in table:
                    if not row:
                        continue
                    cells = [c.strip() for c in row if c and c.strip()]
                    if len(cells) >= 2:
                        pairs = []
                        for i in range(0, len(cells) - 1, 2):
                            pairs.append(f"{cells[i]}: {cells[i+1]}")
                        table_text_parts.append(" | ".join(pairs))
                    elif len(cells) == 1:
                        table_text_parts.append(cells[0])

            table_text = "\n".join(table_text_parts)

            # Combine table text and remaining page text
            combined = f"{table_text}\n\n{remaining_text}".strip()

            if combined:
                documents.append(
                    Document(
                        page_content=combined,
                        metadata={"source": pdf_path, "page": page_num, "section": "body_content"},
                    )
                )

    return documents
