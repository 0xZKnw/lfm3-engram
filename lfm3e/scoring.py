"""Answer matching shared by the runs from D14 on.

Earlier runs counted an answer as right when an accepted alias appeared anywhere in the output as a substring, so
'pol' matched 'police'. Here an alias must appear as whole words, on text with accents folded and case ignored."""
import re
import unicodedata


def fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()


def matches(output, golds, min_len=2):
    text = fold(output)
    for g in golds:
        g = fold(g).strip()
        if len(g) >= min_len and re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", text):
            return True
    return False
