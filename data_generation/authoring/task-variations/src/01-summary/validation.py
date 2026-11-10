"""Strict validation for the seven-section summary contract."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List


EXPECTED_TAGS = (
    "rules_summary",
    "input_generation",
    "variation_guidance",
    "train_test_shift",
    "solution_steps",
    "key_insight",
    "puzzle_concepts",
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    actual_tags: List[str]
    errors: List[str]
    empty_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "expected_tags": list(EXPECTED_TAGS),
            "actual_tags": self.actual_tags,
            "empty_tags": self.empty_tags,
            "errors": self.errors,
        }


def validate_summary_xml(response: str) -> ValidationResult:
    errors: List[str] = []
    actual_tags: List[str] = []
    empty_tags: List[str] = []

    if not isinstance(response, str) or not response.strip():
        return ValidationResult(False, [], ["Response is empty."], [])

    if "```" in response:
        errors.append("Markdown code fences are not allowed.")
    if re.search(r"<\?(?:xml|[^>]+)\?>|<!--|<!DOCTYPE", response, re.IGNORECASE):
        errors.append("XML declarations, processing instructions, comments, and doctypes are not allowed.")

    for tag in EXPECTED_TAGS:
        opening = len(re.findall(r"<%s(?:\s[^>]*)?>" % re.escape(tag), response))
        closing = response.count("</%s>" % tag)
        if opening != 1 or closing != 1:
            errors.append("Section <%s> must have exactly one opening and closing tag." % tag)

    try:
        root = ET.fromstring("<summary_root>" + response + "</summary_root>")
    except ET.ParseError as exc:
        errors.append("Response is not well-formed XML: %s" % exc)
        return ValidationResult(False, actual_tags, errors, empty_tags)

    actual_tags = [child.tag for child in root]
    if actual_tags != list(EXPECTED_TAGS):
        errors.append("Sections are missing, extra, duplicated, or out of order.")

    if (root.text or "").strip():
        errors.append("Text appears before the first XML section.")
    for child in root:
        if (child.tail or "").strip():
            errors.append("Text appears outside the required XML sections.")
            break
        if child.attrib:
            errors.append("Attributes are not allowed on <%s>." % child.tag)
        if list(child):
            errors.append("Nested XML tags are not allowed inside <%s>." % child.tag)
        if not "".join(child.itertext()).strip():
            empty_tags.append(child.tag)
    if empty_tags:
        errors.append("Required sections must not be empty.")

    return ValidationResult(not errors, actual_tags, errors, empty_tags)
