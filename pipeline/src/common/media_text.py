from __future__ import annotations

import re


TOPICS_LINE_RE = re.compile(
    r"^TEME:[^\n]*(?:\n\n?|\Z)",
    flags=re.IGNORECASE,
)
ARTIFACT_CHARS = "_^"
ARTIFACT_TOKEN_RE = re.compile(r"[_\^]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
SENTENCE_START_RE = re.compile(r"\b[A-ZČŠŽ][a-zčšž]{3,}\b")
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def normalize_newlines(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _is_probable_artifact_token(token: str) -> bool:
    candidate = token.strip()
    if not candidate:
        return False

    artifact_count = sum(char in ARTIFACT_CHARS for char in candidate)
    alpha_count = sum(char.isalpha() for char in candidate)
    upper_count = sum(char.isupper() for char in candidate)
    digit_count = sum(char.isdigit() for char in candidate)
    other_count = len(candidate) - alpha_count - digit_count

    if artifact_count == 0 and "__" not in candidate and "^^" not in candidate:
        return False
    if artifact_count >= 1 and len(candidate) >= 4:
        return True
    if artifact_count >= 1 and alpha_count == 0:
        return True
    if artifact_count >= 2:
        return True
    if "__" in candidate or "^^" in candidate:
        return True
    if upper_count >= 2 and artifact_count >= 1:
        return True
    if len(candidate) >= 6 and alpha_count <= 2 and other_count >= 3:
        return True
    if len(candidate) >= 10 and artifact_count >= 1 and alpha_count / max(len(candidate), 1) < 0.45:
        return True
    return False


def _clean_artifact_token_runs(line: str) -> str:
    tokens = line.split()
    if not tokens:
        return ""

    cleaned: list[str] = []
    run: list[str] = []

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        run_len = len(run)
        at_edge = not cleaned or len(cleaned) == 0
        if run_len >= 2 or (at_edge and run_len >= 1):
            run = []
            return
        cleaned.extend(run)
        run = []

    for token in tokens:
        if _is_probable_artifact_token(token):
            run.append(token)
            continue
        flush_run()
        cleaned.append(token)

    if run:
        if len(run) < 3 and cleaned:
            cleaned.extend(run)

    return " ".join(cleaned)


def _trim_artifact_prefix(text: str) -> str:
    head = text[:240]
    if sum(char in ARTIFACT_CHARS for char in head) < 4:
        return text

    for match in SENTENCE_START_RE.finditer(head):
        prefix = head[:match.start()]
        if len(prefix) >= 20 and sum(char in ARTIFACT_CHARS for char in prefix) >= 2:
            return text[match.start():].lstrip(" ;:,-")
    return text


def _trim_artifact_suffix(text: str) -> str:
    tail = text[-240:]
    if sum(char in ARTIFACT_CHARS for char in tail) < 4:
        return text

    last_sentence_end = None
    for match in SENTENCE_END_RE.finditer(text):
        last_sentence_end = match.end()
    if last_sentence_end is None or last_sentence_end >= len(text):
        return text

    suffix = text[last_sentence_end:]
    if len(suffix) >= 12 and sum(char in ARTIFACT_CHARS for char in suffix) >= 2:
        return text[:last_sentence_end].rstrip()
    return text


def clean_obvious_ocr_artifacts(text: str) -> str:
    """
    Conservatively remove obvious OCR/layout garbage without changing the
    default benchmark path.

    The intent is to strip long runs of artifact-heavy tokens such as:
    `_W_w^ "%__. ___wwww * >^ * __________BKKfB^Mh^-_`
    while leaving ordinary punctuation and article prose intact.
    """
    normalized = normalize_newlines(text)
    cleaned_lines = []
    artifact_touched = False
    for raw_line in normalized.split("\n"):
        if not ARTIFACT_TOKEN_RE.search(raw_line):
            cleaned_lines.append(raw_line.rstrip())
            continue

        artifact_touched = True
        line = MULTISPACE_RE.sub(" ", raw_line.strip())
        if not line:
            cleaned_lines.append("")
            continue
        line = _clean_artifact_token_runs(line)
        line = MULTISPACE_RE.sub(" ", line).strip()
        if line:
            cleaned_lines.append(line)
        else:
            cleaned_lines.append("")

    cleaned = "\n".join(cleaned_lines).strip()
    if artifact_touched:
        cleaned = EXCESS_NEWLINES_RE.sub("\n\n", cleaned)
        cleaned = _trim_artifact_prefix(cleaned)
        cleaned = _trim_artifact_suffix(cleaned)
    return cleaned.strip()


def build_model_text(title: str, combined_text: str, clean_obvious_artifacts: bool = False) -> str:
    """
    Canonical modeling text for media articles: title + body only.

    The intermediate TEME/topics line is preserved elsewhere as metadata for
    stratification or audit, but must not be passed to the model.
    """
    title_text = str(title or "").strip()
    text = normalize_newlines(combined_text).strip()

    if title_text and text.startswith(title_text):
        text = text[len(title_text):].lstrip("\n")

    text = TOPICS_LINE_RE.sub("", text, count=1).strip()
    if clean_obvious_artifacts:
        text = clean_obvious_ocr_artifacts(text)

    if title_text and text:
        return f"{title_text}\n\n{text}"
    if title_text:
        return title_text
    return text
