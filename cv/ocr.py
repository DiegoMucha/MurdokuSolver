import json
import re

import cv2
import pytesseract

Y1 = 0
Y2 = 1080
X1 = 0
X2 = 1020

MIN_CONFIDENCE = 50
VICTIM_NAME = "Josh"

OBJECT_NAMES = {
    "shelf": "shelf",
    "shelves": "shelf",
    "plant": "plant",
    "plants": "plant",
    "carpet": "carpet",
    "carpets": "carpet",
    "chair": "chair",
    "chairs": "chair",
    "table": "table",
    "tables": "table",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

ORDINAL_GRID_INDEXES = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
}

NAME_PATTERN = r"[A-Z][A-Za-z'-]*"
OBJECT_PATTERN = r"shel(?:f|ves)|plants?|carpets?|chairs?|tables?"
ROOM_PATTERN = r"yellow|blue|green|orange"
OWNER_PATTERN = re.compile(
    rf"\b(?P<person>{NAME_PATTERN})\s+"
    r"(?=(?:(?:She|He|They)\s+)?(?:was|Was|were|Were|is|Is)\b|"
    r"(?:The|the)\s+victim\b)",
)


def crop_clue_region(image):
    return image[Y1:Y2, X1:X2]


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def extract_words(processed_image):
    data = pytesseract.image_to_data(
        processed_image,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []

    for i, text in enumerate(data["text"]):
        text = text.strip()

        if not text:
            continue

        confidence = float(
            data["conf"][i]
        )

        if confidence >= MIN_CONFIDENCE:
            words.append({
                "text": text,
                "confidence": confidence
            })

    return words


def _normalise_object(value):
    return OBJECT_NAMES[value.lower()]


def _normalise_number(value):
    value = value.lower()
    if value.isdigit():
        return int(value)
    if value in ORDINAL_GRID_INDEXES:
        return ORDINAL_GRID_INDEXES[value]
    return NUMBER_WORDS[value]


def _split_clues_by_person(text):
    """Split flattened Tesseract text into (person, clue text) pairs."""
    matches = [
        match
        for match in OWNER_PATTERN.finditer(text)
        if match.group("person").lower() not in {"she", "he", "they"}
    ]
    sections = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        clue_text = text[start:end].strip(" .")
        clue_text = re.sub(
            r"^(?:she|he|they)\s+",
            "",
            clue_text,
            flags=re.IGNORECASE,
        )
        sections.append((match.group("person").title(), clue_text))

    return sections


def _parse_person_clues(person, clue_text):
    """Convert one person's natural-language clue text to normalized rules."""
    found_clues = []

    def add(position, relation, argument):
        clue = [person, relation, argument]
        if not any(existing == clue for _, existing in found_clues):
            found_clues.append((position, clue))

    # Negative rules must be checked before their positive equivalents.
    for match in re.finditer(
        rf"\bnot\s+(?:in|inside)\s+(?:the\s+)?({ROOM_PATTERN})\s+room\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "not in room", match.group(1).lower())

    for match in re.finditer(
        rf"\bnot\s+(?:besides?|next\s+to|adjacent\s+to)\s+(?:a|an|the)?\s*({OBJECT_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "not beside", _normalise_object(match.group(1)))

    for match in re.finditer(
        rf"\b(?:in|inside)\s+(?:the\s+)?({ROOM_PATTERN})\s+room\b",
        clue_text,
        re.IGNORECASE,
    ):
        prefix = clue_text[max(0, match.start() - 5):match.start()].lower()
        if "not" not in prefix:
            add(match.start(), "in room", match.group(1).lower())

    for match in re.finditer(
        rf"\b(?:sitting\s+)?(?:on|on\s+top\s+of)\s+(?:a|an|the)?\s*({OBJECT_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "on top", _normalise_object(match.group(1)))

    for match in re.finditer(
        rf"\b(?:besides?|next\s+to|adjacent\s+to)\s+(?:a|an|the)?\s*({OBJECT_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        prefix = clue_text[max(0, match.start() - 5):match.start()].lower()
        if "not" not in prefix:
            add(match.start(), "beside", _normalise_object(match.group(1)))

    number_pattern = r"\d+|" + "|".join(
        [*NUMBER_WORDS, *ORDINAL_GRID_INDEXES]
    )

    for match in re.finditer(
        rf"\b(?:in|on)\s+(?:the\s+)?({number_pattern})(?:st|nd|rd|th)?\s+row\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "row", _normalise_number(match.group(1)))

    for match in re.finditer(
        rf"\b(?:in|on)\s+(?:the\s+)?({number_pattern})(?:st|nd|rd|th)?\s+column\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "column", _normalise_number(match.group(1)))

    for match in re.finditer(
        rf"\b(?:in\s+the\s+)?same\s+room\s+as\s+({NAME_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "same room", match.group(1).title())

    for match in re.finditer(
        rf"\b(?:in\s+a\s+)?different\s+room\s+(?:from|than)\s+({NAME_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "different room", match.group(1).title())

    for match in re.finditer(
        rf"\b(?:besides?|next\s+to|adjacent\s+to)\s+({NAME_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        if match.group(1).lower() not in {"a", "an", "the", *OBJECT_NAMES}:
            add(match.start(), "beside", match.group(1).title())

    direction_pattern = re.compile(
        rf"\b(?:exactly\s+)?(?:(?P<leading_distance>{number_pattern})\s+"
        rf"(?:rows?|columns?)\s+)?(?P<direction>north|south|east|west)\s+of\s+"
        rf"(?P<target>{NAME_PATTERN})"
        rf"(?:\s+(?:by|at\s+a\s+distance\s+of)\s+(?P<trailing_distance>{number_pattern}))?\b",
        re.IGNORECASE,
    )
    for match in direction_pattern.finditer(clue_text):
        argument = match.group("target").title()
        distance = match.group("leading_distance") or match.group("trailing_distance")
        clue = [person, f"{match.group('direction').lower()} of", argument]
        if distance is not None:
            clue.append(_normalise_number(distance))
        if not any(existing == clue for _, existing in found_clues):
            found_clues.append((match.start(), clue))

        remaining_text = clue_text[match.end():]
        another_area = re.match(
            r"\s+in\s+(?:another|a\s+different)\s+(?:area|room)\b",
            remaining_text,
            re.IGNORECASE,
        )
        if another_area:
            add(match.end(), "different room", argument)

    for match in re.finditer(
        rf"\b(?:in\s+)?(?:another|a\s+different)\s+(?:area|room)\s+"
        rf"(?:from|than|of)\s+({NAME_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "different room", match.group(1).title())

    for match in re.finditer(
        rf"\balone\s+with\s+(?:the\s+)?({NAME_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    ):
        add(match.start(), "alone with", match.group(1).lower())

    alone_match = re.search(
        r"\b(?:was\s+)?alone\b(?!\s+with)",
        clue_text,
        re.IGNORECASE,
    )
    if alone_match:
        add(alone_match.start(), "alone", None)

    found_clues.sort(key=lambda item: item[0])
    return [clue for _, clue in found_clues]


def _exclusive_objects(clue_text):
    """Return objects that the clue says only this person can be on."""
    matches = re.finditer(
        rf"\b(?:the\s+)?only\s+person\s+(?:sitting\s+)?"
        rf"(?:on|on\s+top\s+of)\s+(?:a|an|the)?\s*({OBJECT_PATTERN})\b",
        clue_text,
        re.IGNORECASE,
    )
    return [_normalise_object(match.group(1)) for match in matches]


def parse_clues(raw_ocr_output):
    """Convert read_clues() output into a JSON-compatible clue dictionary."""
    if isinstance(raw_ocr_output, dict):
        text = raw_ocr_output.get("text", "")
    elif isinstance(raw_ocr_output, str):
        text = raw_ocr_output
    else:
        raise TypeError("raw_ocr_output must be a dictionary or string")

    sections = _split_clues_by_person(text)
    clues_by_person = {person: [] for person, _ in sections}
    exclusive_rules = []

    for person, clue_text in sections:
        if person != VICTIM_NAME:
            clues_by_person[person].extend(_parse_person_clues(person, clue_text))

        for object_name in _exclusive_objects(clue_text):
            exclusive_rules.append((person, object_name))

    for only_person, object_name in exclusive_rules:
        for person in clues_by_person:
            if person == only_person:
                continue

            negative_clue = [person, "not on top", object_name]
            if negative_clue not in clues_by_person[person]:
                clues_by_person[person].append(negative_clue)

    clues = [
        clue
        for person_clues in clues_by_person.values()
        for clue in person_clues
    ]

    return {"clues": clues}


def clues_to_json(raw_ocr_output, indent=2):
    """Return parsed OCR clues serialized as JSON text."""
    return json.dumps(
        parse_clues(raw_ocr_output),
        ensure_ascii=False,
        indent=indent,
    )


def read_clues(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    clue_region = crop_clue_region(image)

    processed = preprocess_for_ocr(
        clue_region
    )

    words = extract_words(
        processed
    )

    full_text = " ".join(
        word["text"]
        for word in words
    )

    return {
        "text": full_text,
    }
