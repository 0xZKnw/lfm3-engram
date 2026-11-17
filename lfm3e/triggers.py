"""When to show a note: the rules that pick names and terms in a text. One copy, used by the experiments and
shipped with the integrated model (this file only needs the standard library).

  - question_spans / pick_spans: capitalised names of a question (D5)
  - cap_runs / find_triggers: precision-first rules, exact titles only, English and French (D6)"""
import re
import unicodedata

STOP = {"what", "who", "which", "where", "when", "why", "how", "in", "is", "was", "the", "of", "for", "a", "an", "by", "does", "did", "do", "on", "to", "at", "from"}


def norm(s):
    """Lower-case, accents folded ('José' -> 'jose'), possessives and punctuation dropped."""
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"['’]s\b", "", s)
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def question_spans(q, max_len=6, general=False):
    """Eligible spans (i, j, surface): contain a capitalised or numeric word that is not the first word.

    general=True is for any prompt, not only a one-sentence question: a capital says nothing on the first word
    of a sentence or of a line, a number alone is not a name, function words ('I', 'The') do not count, and a
    name does not run across two sentences."""
    words = q.split()
    cleaned = [norm(w) for w in words]
    if general:
        starts, pos = [], 0
        for k, w in enumerate(words):
            at = q.index(w, pos)
            starts.append(k == 0 or "\n" in q[pos:at] or words[k - 1][-1:] in ".!?:;")
            pos = at + len(w)
        signal = [not starts[k] and words[k][:1].isupper() and cleaned[k] not in CAP_STOP and cleaned[k] not in FR_STOP for k in range(len(words))]
    else:
        starts = [False] * len(words)
        signal = [k > 0 and (words[k][:1].isupper() or words[k][:1].isdigit()) for k in range(len(words))]
    spans = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + max_len, len(words)) + 1):
            parts = [c for c in cleaned[i:j] if c]
            if not parts:
                continue
            if not any(signal[i:j]) or any(starts[i + 1 : j]):  # (general) a name does not run across two sentences
                continue
            if j - i == 1 and cleaned[i] in STOP:
                continue
            if general and j - i == 1 and (len(cleaned[i]) < 3 or cleaned[i] in CAP_STOP or cleaned[i] in FR_STOP):
                continue
            spans.append((i, j, " ".join(parts)))
    return spans


def pick_spans(q, cands, max_spans=2, general=False):
    """Longest non-overlapping eligible spans that have candidates."""
    found = sorted((s for s in question_spans(q, general=general) if s[2] in cands), key=lambda s: (s[0] - s[1], s[0]))
    chosen, used = [], set()
    for i, j, surface in found:
        if used.isdisjoint(range(i, j)):
            chosen.append((i, j, surface))
            used.update(range(i, j))
        if len(chosen) >= max_spans:
            break
    return chosen


CAP_STOP = {
    "the", "this", "that", "these", "those", "we", "it", "in", "if", "so", "but", "and", "then", "since", "because", "however",
    "therefore", "thus", "let", "given", "for", "as", "to", "now", "first", "second", "third", "next", "also", "they", "he", "she",
    "there", "what", "who", "which", "where", "when", "why", "how", "is", "was", "are", "were", "answer", "question", "note", "yes",
    "no", "a", "an", "i", "you", "your", "my", "our", "its", "his", "her", "their", "on", "at", "by", "from", "with", "or", "not",
    "both", "each", "all", "some", "one", "two", "three", "here", "step", "option", "options", "final", "reference", "user", "okay",
    "ok", "hmm", "wait", "actually", "maybe", "perhaps", "after", "before", "while", "during", "according", "based", "need", "check",
}
CONNECT = {"of", "the", "and", "de", "la", "von", "van", "in", "for", "on", "at", "to", "del", "di", "da", "du", "le", "des", "les", "et", "en", "au", "aux", "d'", "l'", "d’", "l’"}
FR_STOP = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou", "à", "a", "en", "dans", "sur", "pour", "par", "avec", "sans",
    "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "où", "ou", "quand", "comment", "combien", "pourquoi", "est", "sont", "était",
    "étaient", "été", "être", "avoir", "ont", "il", "elle", "ils", "elles", "on", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur",
    "leurs", "plus", "moins", "très", "ne", "pas", "se", "si", "mais", "donc", "car", "comme", "entre", "vers", "chez", "depuis", "pendant",
    "réponse", "question", "note", "oui", "non", "je", "tu", "nous", "vous", "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre",
    "premier", "première", "dernier", "dernière", "tout", "tous", "toute", "toutes", "fait", "faire", "y", "lors", "après", "avant",
    "l", "d", "j", "n", "s", "c", "m", "t", "qu",
}
FUNC = CAP_STOP | CONNECT | FR_STOP | {
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may", "might",
    "must", "than", "then", "more", "most", "less", "least", "very", "such", "any", "other", "another", "same", "different", "between",
    "into", "about", "over", "under", "per", "following", "which", "whose", "whom", "use", "used", "using", "find", "get", "give",
}


def cap_runs(text):
    """Runs of capitalised words ('Kiss and Tell', 'Shirley Temple'), each with the up to three words that
    precede it ('guerre de' before 'Cent Ans'). A single capitalised word that starts a sentence is dropped: the
    capital says nothing there."""
    text = re.sub(r"\b([ldjmnstcLDJMNSTC]|qu|Qu)['’](?=\w)", r"\1' ", text)  # l'Allemagne -> l' Allemagne
    toks = re.findall(r"[^\W_][\w'’\-]*|[.!?\n:;]", text)
    runs, i, sentence_start = [], 0, True
    while i < len(toks):
        t = toks[i]
        if t in ".!?\n:;":
            sentence_start, i = True, i + 1
            continue
        if t[0].isupper():
            j, words, at_start = i + 1, [t], sentence_start
            while j < len(toks):
                if toks[j][0].isupper():
                    words.append(toks[j])
                    j += 1
                elif toks[j].lower() in CONNECT and j + 1 < len(toks) and toks[j + 1][0].isupper():
                    words += [toks[j], toks[j + 1]]
                    j += 2
                else:
                    break
            if not (len(words) == 1 and (at_start or words[0].lower() in CAP_STOP or words[0].lower() in FR_STOP)):
                left = []
                for k in range(i - 1, max(i - 4, -1), -1):
                    if toks[k] in ".!?\n:;":
                        break
                    left.insert(0, toks[k])
                runs.append((words, left))
            i = j
        else:
            i += 1
        sentence_start = False
    return runs


def find_triggers(text, exists, freq, ntok, max_concepts=4):
    """Surfaces to look up, as (surface, kind). Precision first: a wrong note costs more than a missing one.

    `exists(surface)`: an article has exactly this title. `freq(word)`: occurrences per million tokens of general
    web text (0 when the tokenizer splits the word, i.e. the word is rare). `ntok(word)`: number of tokens.

      - names: runs of at least two capitalised words that are an article title; a single word only when the
        tokenizer needs 3 tokens or more for it ('Pleiospilos' yes; 'Andy', 'Multiply', 'Switzerland' no)
      - concepts: 2-4 word terms that are an article title and contain an uncommon word ('center of mass' no,
        'pyridinium chlorochromate' yes); one word only when it is long and split into 3 tokens or more"""
    out, seen = [], set()

    def add(surface, kind):
        if surface and surface not in seen:
            seen.add(surface)
            out.append((surface, kind))

    def match(words):
        """Longest sub-span that is a title, then the same on what is left on each side."""
        for size in range(len(words), 0, -1):
            for a in range(0, len(words) - size + 1):
                sub = words[a : a + size]
                if sub[0] in CONNECT or sub[-1] in CONNECT:
                    continue
                surface = norm(" ".join(sub))
                if size == 1 and (len(surface) < 5 or " " in surface or surface in CAP_STOP or surface in FR_STOP or ntok(sub[0]) < 3):
                    continue
                if size > 1 and (len(surface.replace(" ", "")) < 5 or max(len(w) for w in surface.split()) < 3):
                    continue
                if exists(surface):
                    match(words[:a])
                    add(surface, "name")
                    match(words[a + size :])
                    return

    for words, left in cap_runs(text):
        for k in range(len(left), 0, -1):  # 'guerre de' + 'Cent Ans', 'battle of' + 'Hastings'
            ext = left[-k:] + words
            if ext[0].lower() in FUNC or len(ext) > 10:
                continue
            surface = norm(" ".join(ext))
            if exists(surface):
                add(surface, "name")
                break
        else:
            match(words[:10])

    text2 = re.sub(r"\b([ldjmnstc]|qu)['’](?=\w)", r"\1 ", text.lower())
    broken = re.sub(r"[.,;:!?()\[\]\n\"*_$\\{}=+<>/|^«»]", " zzzbreak ", re.sub(r"['’]s?(?![^\W\d_])", "", text2))
    low = re.findall(r"[^\W\d_][^\W\d_\-]+|[^\W\d_]", broken)
    found = []
    for size in (5, 4, 3, 2, 1):
        for a in range(0, len(low) - size + 1):
            sub = low[a : a + size]
            if "zzzbreak" in sub or sub[0] in FUNC or sub[-1] in FUNC or any(len(w) < 2 for w in sub):
                continue
            content = [w for w in sub if w not in FUNC]
            rarest = min(freq(w) for w in content)
            if size == 1 and (len(sub[0]) < 8 or ntok(sub[0]) < 3):
                continue
            if size > 1 and rarest > 20:
                continue
            surface = norm(" ".join(sub))
            if len(surface) < 6 or surface in seen or any(surface in f for f, _ in found) or any(surface in n for n in seen):
                continue
            if exists(surface):
                found.append((surface, rarest))
    for surface, _ in sorted(found, key=lambda x: (-len(x[0].split()), x[1]))[:max_concepts]:
        add(surface, "concept")
    return out
