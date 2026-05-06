from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib


DISCUSSION_HINT_WORDS = ("讨论", "交流", "商量")
HELP_HINT_WORDS = ("帮", "帮助", "协助")
EVALUATION_HINT_WORDS = ("觉得", "认为", "评价", "怎么看", "对吗", "对不对", "是否正确", "答案", "回答")
MAX_TARGETS = 3
SINGULAR_ADDRESSEE_PREFIXES = ("你", "，你", ",你", "同学你", "同学，你", "同学,你")
PLURAL_ADDRESSEE_PREFIXES = ("你们", "，你们", ",你们", "你俩", "，你俩", ",你俩")
COORDINATION_SEGMENTS = {"", "，", ",", "、", "和", "及", "与", "跟", "，和", ",和", "，及", ",及", "，与", ",与"}
ADDRESS_FILLER_WORDS = ("啊", "呀", "呢", "呃", "额", "哦", "诶", "欸")


@dataclass(frozen=True)
class PersonAlias:
    canonical_name: str
    alias: str


def load_people_config(config_path: str | Path) -> list[PersonAlias]:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    aliases: list[PersonAlias] = []
    for person in data["people"]:
        canonical_name = person["name"].strip()
        for alias in person.get("aliases", []):
            alias = str(alias).strip()
            if alias:
                aliases.append(PersonAlias(canonical_name=canonical_name, alias=alias))

    aliases.sort(key=lambda item: len(item.alias), reverse=True)
    return aliases


def find_candidate_names(text: str, people_aliases: list[PersonAlias]) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for item in people_aliases:
        start = 0
        while True:
            index = text.find(item.alias, start)
            if index == -1:
                break
            end = index + len(item.alias)
            if not any(index < used_end and end > used_start for used_start, used_end in occupied):
                matches.append((index, end, item.canonical_name))
                occupied.append((index, end))
            start = index + 1

    ordered_names: list[str] = []
    seen: set[str] = set()
    for _, _, name in sorted(matches, key=lambda item: item[0]):
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)
    return ordered_names


def build_target_feature(text: str, candidate_name: str) -> str:
    index = text.find(candidate_name)
    if index == -1:
        return f"句子:{text}\n候选:{candidate_name}\n位置:missing"

    left = text[max(0, index - 8):index]
    right = text[index + len(candidate_name): index + len(candidate_name) + 8]
    marked_text = text.replace(candidate_name, "[TARGET]", 1)
    return (
        f"句子:{text}\n"
        f"候选:{candidate_name}\n"
        f"标注句子:{marked_text}\n"
        f"左侧:{left}\n"
        f"右侧:{right}"
    )


def _extract_discussion_targets(text: str, candidate_names: list[str]) -> list[str]:
    discussion_positions = [text.find(word) for word in DISCUSSION_HINT_WORDS if word in text]
    if not discussion_positions:
        return []

    discussion_index = min(discussion_positions)
    before_discussion = [name for name in candidate_names if text.find(name) != -1 and text.find(name) < discussion_index]
    if len(before_discussion) >= 2:
        return before_discussion[:MAX_TARGETS]

    selected: list[str] = []
    for candidate in candidate_names:
        if candidate in text:
            selected.append(candidate)

    if len(selected) < 2:
        return []
    return selected[:MAX_TARGETS]


def _extract_help_actor(text: str, candidate_names: list[str]) -> list[str]:
    if not any(word in text for word in HELP_HINT_WORDS):
        return []

    if not candidate_names:
        return []

    help_index = min(text.find(word) for word in HELP_HINT_WORDS if word in text)
    actors = [name for name in candidate_names if text.find(name) != -1 and text.find(name) < help_index]
    if actors:
        return actors[:1]
    return candidate_names[:1]


def _collect_coordinated_names(text: str, candidate_names: list[str], end_index: int) -> list[str]:
    positions = [(name, text.find(name)) for name in candidate_names[: end_index + 1]]
    positions = [(name, start) for name, start in positions if start != -1]
    if not positions:
        return []

    collected = [positions[-1][0]]
    current_start = positions[-1][1]

    for name, start in reversed(positions[:-1]):
        segment = text[start + len(name):current_start].replace(" ", "")
        if segment in COORDINATION_SEGMENTS:
            collected.append(name)
            current_start = start
            continue
        break

    collected.reverse()
    return collected[:MAX_TARGETS]


def _extract_direct_addressee(text: str, candidate_names: list[str]) -> list[str]:
    if not candidate_names:
        return []

    for candidate_index, candidate_name in enumerate(candidate_names):
        name_start = text.find(candidate_name)
        if name_start == -1:
            continue
        suffix = text[name_start + len(candidate_name): name_start + len(candidate_name) + 8]
        normalized_suffix = suffix.replace(" ", "")
        if normalized_suffix.startswith("同学"):
            normalized_suffix = normalized_suffix[2:]
            normalized_suffix = normalized_suffix.lstrip("，,。")
            while normalized_suffix.startswith(ADDRESS_FILLER_WORDS):
                normalized_suffix = normalized_suffix[1:]
            normalized_suffix = normalized_suffix.lstrip("，,。")
        if normalized_suffix.startswith(PLURAL_ADDRESSEE_PREFIXES):
            coordinated_names = _collect_coordinated_names(text, candidate_names, candidate_index)
            if coordinated_names:
                return coordinated_names
            return [candidate_name]
        if normalized_suffix.startswith(SINGULAR_ADDRESSEE_PREFIXES):
            return [candidate_name]
    return []


def _extract_leading_evaluation_speaker(text: str, candidate_names: list[str]) -> list[str]:
    if len(candidate_names) < 2:
        return []

    first_name = candidate_names[0]
    if not text.startswith(first_name):
        return []

    suffix = text[len(first_name): len(first_name) + 2]
    if not any(mark in suffix for mark in ("，", ",", "。")):
        return []

    if any(word in text for word in EVALUATION_HINT_WORDS):
        return [first_name]
    return []


def apply_high_precision_rules(text: str, candidate_names: list[str]) -> list[str]:
    discussion_targets = _extract_discussion_targets(text, candidate_names)
    if discussion_targets:
        return discussion_targets

    help_targets = _extract_help_actor(text, candidate_names)
    if help_targets:
        return help_targets

    direct_addressee_targets = _extract_direct_addressee(text, candidate_names)
    if direct_addressee_targets:
        return direct_addressee_targets

    leading_evaluation_speaker = _extract_leading_evaluation_speaker(text, candidate_names)
    if leading_evaluation_speaker:
        return leading_evaluation_speaker

    return []


class TargetExtractor:
    def __init__(
        self,
        model_path: str | Path,
        people_config_path: str | Path,
        threshold: float = 0.5,
        max_targets: int = MAX_TARGETS,
    ) -> None:
        self.model = joblib.load(model_path)
        self.people_aliases = load_people_config(people_config_path)
        self.threshold = threshold
        self.max_targets = max_targets

    def predict(self, text: str) -> dict[str, object]:
        candidate_names = find_candidate_names(text, self.people_aliases)
        if not candidate_names:
            return {
                "text": text,
                "targets": [],
                "candidates": [],
                "rule_applied": False,
            }

        rule_targets = apply_high_precision_rules(text, candidate_names)
        if rule_targets:
            return {
                "text": text,
                "targets": rule_targets[: self.max_targets],
                "candidates": candidate_names,
                "rule_applied": True,
            }

        features = [build_target_feature(text, name) for name in candidate_names]
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
            selected = [
                name
                for name, score in zip(candidate_names, scores)
                if float(score) >= self.threshold
            ]
        else:
            decisions = self.model.decision_function(features)
            selected = [
                name
                for name, score in zip(candidate_names, decisions)
                if float(score) >= 0.0
            ]

        return {
            "text": text,
            "targets": selected[: self.max_targets],
            "candidates": candidate_names,
            "rule_applied": False,
        }
