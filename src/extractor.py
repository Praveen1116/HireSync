import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Cleans raw text by removing extra spaces, line breaks,
    and special symbols while preserving skill characters like +, #, ., /
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\+\#\.\/\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_pdf(pdf_source) -> str:
    """
    Extracts and cleans all text from a PDF file.
    
    Args:
        pdf_source: A file path string or a file-like object (e.g. Streamlit upload).
        
    Returns:
        Cleaned text extracted from the PDF pages.
    """
    reader = PdfReader(pdf_source)
    extracted_pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text)

    full_raw_text = " ".join(extracted_pages)
    return clean_text(full_raw_text)

