from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.cluster.hierarchy import linkage


MULTILINGUAL_STOPWORD_LANGUAGES = [
    "english",
    "french",
    "spanish",
    "italian",
    "german",
    "dutch",
]

DOMAIN_STOP_WORDS = [
    "hotel",
    "room",
    "rooms",
    "stay",
    "stayed",
    "booking",
    "booked",
    "check",
    "night",
    "day",
    "time",
    "area",
    "place",
    "city",
    "walk",
    "minutes",
    "people",
    "arrival",
    "asked",
    "told",
    "got",
    "pay",
    "work",
    "located",
    "near",
    "away",
    "floor",
    "building",
    "way",
    "morning",
]

LOCATION_STOP_WORDS = [
    "amsterdam",
    "barcelona",
    "london",
    "milan",
    "paris",
    "vienna",
    "netherlands",
    "spain",
    "france",
    "italy",
    "austria",
    "eiffel",
    "oxford",
    "gogh",
    "montmartre",
    "milano",
    "centre",
    "center",
    "street",
    "europe",
    "european",
]

ZERO_SHOT_MAJOR_TOPICS = [
    "Staff Service",
    "Room Comfort & Quality",
    "Cleanliness",
    "Location & Accessibility",
    "Breakfast & Food",
    "Bathroom & Shower Experience",
    "Noise & Sleep Disturbance",
    "Facilities & Amenities",
    "Value for Money",
    "Maintenance & Room Condition",
]

SPLITTER_LIST = [". ", "! ", "? ", "; "]
MIN_WORDS = 3


def ensure_stopwords_downloaded() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def build_multilingual_stop_words(
    extra_stop_words: list[str] | None = None,
) -> list[str]:
    ensure_stopwords_downloaded()

    collected: list[str] = []
    for language in MULTILINGUAL_STOPWORD_LANGUAGES:
        collected.extend(stopwords.words(language))

    collected.extend(DOMAIN_STOP_WORDS)
    collected.extend(LOCATION_STOP_WORDS)

    if extra_stop_words:
        collected.extend(extra_stop_words)

    return sorted(set(collected))


def split_and_update_indices(
    text_list: list[str],
    index_list: list[int],
    split_list: list[str] | None = None,
) -> tuple[list[str], list[int]]:
    delimiters = split_list or SPLITTER_LIST
    new_texts: list[str] = []
    new_indices: list[int] = []

    for text, idx in zip(text_list, index_list):
        fragments = [text]
        for separator in delimiters:
            expanded: list[str] = []
            for fragment in fragments:
                expanded.extend(fragment.split(separator))
            fragments = expanded
        fragments = [fragment for fragment in fragments if fragment.strip()]
        new_texts.extend(fragments)
        new_indices.extend([idx] * len(fragments))

    return new_texts, new_indices


def assign_season(month: object) -> object:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return None


def safe_linkage(matrix: object):
    non_negative = np.clip(np.asarray(matrix), 0, None)
    return linkage(non_negative, "ward")


def make_safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")
