# processors.py
import io
import re
from collections import Counter
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import pdf2image
except Exception:
    pdf2image = None

try:
    from PIL import Image
except Exception:
    Image = None

# OCR libs
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# tokenizers
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str):
    t = text.replace("\r\n", "\n")
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    return t.strip()

def md_from_text(text: str):
    # minimal: wrap paragraphs with double newlines
    return clean_text(text)

def extract_sentences(text: str):
    return sent_tokenize(text)

def top_tokens(text: str, top_k: int = 200):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    cnt = Counter(tokens)
    return cnt.most_common(top_k)

def cooccurrence_pairs(sentences, top_terms, window_size=1):
    top_set = set([t for t, _ in top_terms])
    edges = Counter()
    for s in sentences:
        toks = [w.lower() for w in word_tokenize(s) if w.isalpha()]
        filtered = [t for t in toks if t in top_set]
        for i, a in enumerate(filtered):
            for j in range(i + 1, min(i + 1 + window_size, len(filtered))):
                b = filtered[j]
                if a != b:
                    key = tuple(sorted((a, b)))
                    edges[key] += 1
    return [{"source": a, "target": b, "weight": w} for (a, b), w in edges.items()]

# --- OCR helpers ---
def pdf_to_text_pytesseract(file_bytes: bytes, pages=None, dpi=300):
    """
    Convert PDF bytes to list of images via pdf2image then OCR with pytesseract.
    pages: list of 1-indexed page numbers or None for all.
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not available in environment.")
    if pdf2image is None:
        raise RuntimeError("pdf2image not available. Install pdf2image and poppler.")
    images = pdf2image.convert_from_bytes(file_bytes, dpi=dpi)
    # pages are 1-indexed from pdf2image conversion; choose subset if pages provided
    selected = images if not pages else [images[p-1] for p in pages if 1 <= p <= len(images)]
    out_text = []
    for img in selected:
        txt = pytesseract.image_to_string(img)
        out_text.append(txt)
    return "\n\n".join(out_text)

def pdf_to_text_easyocr(file_bytes: bytes, pages=None, lang_list=['en'], dpi=300):
    """
    Convert PDF bytes to images then OCR with easyocr.Reader
    """
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not available in environment.")
    if pdf2image is None:
        raise RuntimeError("pdf2image not available. Install pdf2image and poppler.")
    reader = easyocr.Reader(lang_list, gpu=False)
    images = pdf2image.convert_from_bytes(file_bytes, dpi=dpi)
    selected = images if not pages else [images[p-1] for p in pages if 1 <= p <= len(images)]
    out_text = []
    for img in selected:
        # easyocr returns list of (bbox, text, confidence) when detail=1, but when detail=0 it's text only
        result = reader.readtext(img, detail=0, paragraph=True)
        if isinstance(result, list):
            out_text.append("\n".join(result))
        else:
            out_text.append(str(result))
    return "\n\n".join(out_text)

def extract_text_from_pdf_bytes(file_bytes: bytes, pages=None, prefer="pytesseract"):
    """
    Convenience wrapper that attempts pdfplumber extraction first, then falls back to chosen OCR.
    pages: list of 1-indexed page numbers (pdfplumber uses 0-index)
    prefer: "pytesseract" or "easyocr"
    """
    # try pdfplumber text extraction
    try:
        if pdfplumber:
            import pdfplumber
            out = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_nums = ( [p-1 for p in pages] if pages else list(range(len(pdf.pages))) )
                for p in page_nums:
                    page = pdf.pages[p]
                    text = page.extract_text()
                    if text and len(text.strip())>20:
                        out.append(text)
                    else:
                        out.append("")  # keep position for OCR fallback
            merged = "\n\n".join(out).strip()
            if merged and len(merged)>50:
                return merged
    except Exception:
        # ignore and fallback to OCR
        pass

    # fallback to OCR
    if prefer == "easyocr" and EASYOCR_AVAILABLE:
        return pdf_to_text_easyocr(file_bytes, pages=pages)
    elif PYTESSERACT_AVAILABLE:
        return pdf_to_text_pytesseract(file_bytes, pages=pages)
    elif EASYOCR_AVAILABLE:
        return pdf_to_text_easyocr(file_bytes, pages=pages)
    else:
        raise RuntimeError("No OCR backends available (pytesseract or easyocr).")
