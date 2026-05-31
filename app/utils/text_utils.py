"""
app/utils/text_utils.py  —  Text cleaning, formatting, and translation helpers.
"""
import re
import logging

log = logging.getLogger("jurisai.utils")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False


def clean_and_format(text: str) -> str:
    """Strip markdown formatting and normalise whitespace/bullets."""
    if not text:
        return ""
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"[_`]", "", text)
    text = re.sub(r"#+", "", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n", text).strip()
    text = re.sub(r"^\s*[-*]\s*", "• ", text, flags=re.M)
    return text.strip()


def safe_translate(text: str, target: str) -> str:
    """Translate *text* to *target* language, falling back to original on error."""
    if not text or target == "en":
        return text
    if not TRANSLATOR_AVAILABLE:
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text) or text
    except Exception:
        return text
