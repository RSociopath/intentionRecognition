from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

from target_extractor import TargetExtractor


QUESTION_HINT_WORDS = (
    "\u63d0\u95ee",
    "\u95ee\u9898",
    "\u7591\u95ee",
    "\u4e3e\u624b",
    "\u591a\u5c11",
    "\u51e0",
    "\u5417",
    "\u5462",
    "\u4e3a\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u600e\u6837",
    "\u662f\u5426",
    "\u5728\u54ea",
    "\u54ea\u91cc",
    "\u54ea\u4e2a",
    "\u54ea\u4e24\u4e2a",
    "\u660e\u767d\u4e86\u5417",
    "\u542c\u61c2\u4e86\u5417",
    "\u7406\u89e3\u4e86\u5417",
    "\uff1f",
    "?",
)
ANSWER_HINT_WORDS = ("\u56de\u7b54", "\u8bf4\u8bf4", "\u89e3\u91ca", "\u8bb2\u8bb2", "\u8865\u5145", "\u4f5c\u7b54", "\u89e3\u7b54", "\u8bf4\u660e")
DISCUSSION_HINT_WORDS = ("\u8ba8\u8bba", "\u63a2\u8ba8", "\u4ea4\u6d41", "\u5546\u91cf")
VOTE_HINT_WORDS = ("\u8868\u51b3", "\u6295\u7968", "\u8d5e\u6210", "\u53cd\u5bf9")
END_CLASS_HINT_WORDS = (
    "\u4e0b\u8bfe",
    "\u8fd9\u8282\u8bfe\u5230\u8fd9\u91cc",
    "\u4eca\u5929\u8fd9\u8282\u8bfe\u5148\u4e0a\u5230\u8fd9\u91cc",
    "\u4eca\u5929\u5c31\u8bb2\u5230\u8fd9\u91cc\uff0c\u4e0b\u8bfe",
    "\u53ef\u4ee5\u4e0b\u8bfe\u4e86",
    "\u540c\u5b66\u4eec\u4e0b\u8bfe",
    "\u8fd9\u8282\u8bfe\u7ed3\u675f",
    "\u51c6\u5907\u4e0b\u8bfe",
)
END_CLASS_FALSE_POSITIVE_PATTERNS = (
    "\u4e0b\u8bfe\u672c",
    "\u770b\u4e0b\u8bfe\u672c",
    "\u7ffb\u4e0b\u8bfe\u672c",
    "\u4e0a\u8bfe",
    "\u5f00\u59cb\u4e0a\u8bfe",
)
START_CLASS_HINT_WORDS = (
    "\u4e0a\u8bfe",
    "\u5f00\u59cb\u4e0a\u8bfe",
    "\u51c6\u5907\u4e0a\u8bfe",
    "\u540c\u5b66\u4eec\u597d\uff0c\u4e0a\u8bfe",
)
SECOND_PERSON_QUESTION_PATTERNS = (
    "\u4f60\u4f1a",
    "\u4f60\u80fd",
    "\u4f60\u53ef\u4ee5",
    "\u4f60\u77e5\u9053",
    "\u4f60\u61c2",
    "\u4f60\u660e\u767d",
    "\u4f60\u7406\u89e3",
    "\u4f60\u8bb0\u5f97",
    "\u4f60\u89c9\u5f97",
    "\u4f60\u544a\u8bc9\u6211",
    "\u8bf7\u4f60",
    "\u5e2e\u6211",
)
DISCOVERY_QUESTION_PATTERNS = (
    "\u4f60\u80fd\u53d1\u73b0\u4ec0\u4e48",
    "\u4f60\u80fd\u770b\u51fa\u4ec0\u4e48",
    "\u4f60\u53d1\u73b0\u4ec0\u4e48",
    "\u4f60\u770b\u51fa\u4ec0\u4e48",
    "\u4ec0\u4e48\u89c4\u5f8b",
)
CONTENT_QUESTION_PATTERNS = (
    "\u7b54\u6848\u662f\u4ec0\u4e48",
    "\u7ed3\u679c\u662f\u4ec0\u4e48",
)
CONTENT_REVIEW_PATTERNS = (
    "\u6211\u4eec\u5b66\u8fc7\u4e86\u4ec0\u4e48\u662f",
    "\u4e4b\u524d\u5b66\u8fc7\u4e86\u4ec0\u4e48\u662f",
    "\u5728\u4e4b\u524d\u7684\u8bfe\u7a0b",
    "\u8fd9\u8282\u8bfe\u6211\u4eec\u8981\u5b66\u4e60",
)
ADDRESSED_ANSWER_PROMPT_PATTERNS = (
    "\u4f60\u6765",
    "\u4f60\u6765\u7b97",
    "\u4f60\u6765\u7b97\u4e00\u4e0b",
    "\u4f60\u6765\u7b97\u4e00\u7b97",
    "\u4f60\u6765\u8ba1\u7b97",
    "\u4f60\u505a",
    "\u4f60\u7b54",
    "\u4f60\u89e3\u91ca",
    "\u4f60\u56de\u7b54",
    "\u4f60\u4f5c\u7b54",
    "\u4e0a\u6765\u505a",
    "\u4e0a\u6765\u7b97",
    "\u6765\u505a\u5427",
    "\u6765\u7b97\u4e00\u4e0b",
    "\u6765\u7b97\u4e00\u7b97",
    "\u6765\u8ba1\u7b97",
    "\u6765\u56de\u7b54",
)
ADDRESSED_QUESTION_PROMPT_PATTERNS = (
    "\u4f60\u8bf4\u8bf4",
    "\u4f60\u8bb2\u8bb2",
    "\u6765\u8bf4\u8bf4",
    "\u6765\u8bb2\u8bb2",
    "\u6765\u8c08\u8c08",
    "\u6765\u8bf4\u4e00\u8bf4",
    "\u6765\u8bb2\u4e00\u8bb2",
    "\u4e00\u8d77\u8bf4\u8bf4",
    "\u4e00\u8d77\u8bb2\u8bb2",
    "\u4e00\u8d77\u8bf4\u4e00\u8bf4",
    "\u4e00\u8d77\u8bb2\u4e00\u8bb2",
)
PLURAL_ADDRESSED_ANSWER_HINT_WORDS = (
    "\u4f60\u4eec",
    "\u4f60\u4fe9",
    "\u4e24\u4e2a",
    "\u4e24\u4f4d",
    "\u4e00\u8d77",
)
VOLUNTEER_QUESTION_HINT_WORDS = ("\u8c01\u80fd", "\u8c01\u6765", "\u8c01\u53ef\u4ee5", "\u8c01\u613f\u610f", "\u54ea\u4f4d\u540c\u5b66", "\u54ea\u4e2a\u540c\u5b66", "\u6709\u6ca1\u6709\u4eba")
REQUEST_QUESTION_PATTERNS = (
    "\u8bf7\u4f60",
    "\u5e2e\u6211",
    "\u9ebb\u70e6\u4f60",
    "\u505a\u4e00\u4e0b",
    "\u505a\u4e00\u505a",
    "\u518d\u505a\u4e00\u4e0b",
    "\u518d\u505a\u4e00\u505a",
    "\u7b97\u4e00\u4e0b",
    "\u7b97\u4e00\u7b97",
    "\u8bf7\u8ba1\u7b97",
    "\u6765\u8ba1\u7b97",
)
REQUEST_QUESTION_PREFIXES = (
    "\u6c42",
    "\u6c42\u51fa",
    "\u6c42\u89e3",
    "\u6c42\u4e0b\u5217",
    "\u6c42\u8bc1",
)
REQUEST_STATEMENT_PATTERNS = (
    "\u6c42\u51fa",
    "\u5c31\u89e3\u51b3\u8fd9\u9053\u9898\u4e86",
    "\u5c31\u80fd\u89e3\u51b3",
)
REFERENCE_OBJECT_SUFFIXES = (
    "\u7684\u95ee\u9898",
    "\u7684\u7591\u95ee",
    "\u7684\u56f0\u60d1",
    "\u7684\u95ee\u9898\u70b9",
    "\u7684\u60f3\u6cd5",
    "\u7684\u601d\u8def",
    "\u7684\u505a\u6cd5",
    "\u7684\u7b54\u6848",
    "\u7684\u610f\u601d",
    "\u540c\u5b66\u7684\u7b54\u6848",
    "\u8bf4\u7684",
    "\u8bb2\u7684",
    "\u5199\u7684",
)
FEEDBACK_CONTINUE_HINT_WORDS = (
    "\u7b54\u5bf9\u4e86",
    "\u7b54\u9519\u4e86",
    "\u8bf4\u5f97\u5bf9",
    "\u8bf4\u5f97\u5f88\u597d",
    "\u56de\u7b54\u5f97\u5f88\u597d",
    "\u56de\u7b54\u5f97\u4e0d\u9519",
    "\u505a\u5f97\u597d",
    "\u505a\u5f97\u4e0d\u9519",
    "\u4e0d\u9519",
    "\u5f88\u597d",
    "\u633a\u5feb",
    "\u633a\u597d",
    "\u633a\u4e0d\u9519",
    "\u5c45\u7136\u53ef\u4ee5",
    "\u8fd8\u53ef\u4ee5\u5440",
)
CONTINUE_HINT_WORDS = (
    "\u516c\u5f0f\u662f",
    "\u5373",
    "\u7b49\u4e8e",
    "\u8868\u793a",
    "\u6211\u4eec\u5df2\u7ecf\u77e5\u9053",
    "\u5b9a\u4e49",
    "\u6982\u5ff5",
    "\u6027\u8d28",
    "\u6211\u4eec\u53d1\u73b0",
    "\u90a3\u4e48",
    "\u5c31\u662f",
    "\u4e5f\u5c31\u662f",
    "\u6709\u5173",
    "\u76f8\u5173",
    "\u5e73\u65b9\u6709\u5173",
    "\u533a\u522b\u5728\u4e8e",
    "\u610f\u5473\u7740",
    "\u8bf4\u660e",
)
TEACHING_LEADIN_PREFIXES = (
    "\u6211\u4eec\u6765",
    "\u63a5\u4e0b\u6765\u6211\u4eec",
    "\u4e0b\u9762\u6211\u4eec",
    "\u73b0\u5728\u6211\u4eec",
    "\u8fd9\u4e00\u9898\u6211\u4eec",
    "\u8fd9\u9053\u9898\u6211\u4eec",
)
TEACHING_TOPIC_HINT_WORDS = (
    "\u5e94\u7528\u573a\u666f",
    "\u5e94\u7528",
    "\u516c\u5f0f",
    "\u6982\u5ff5",
    "\u65b9\u6cd5",
    "\u6b65\u9aa4",
    "\u6027\u8d28",
    "\u4f8b\u9898",
    "\u7b49\u6bd4\u6570\u5217",
    "\u7684\u7ed3\u679c",
)
TEACHING_OBJECTIVE_PREFIX_PATTERNS = (
    "\u6211\u4eec\u8fd9\u8282\u8bfe\u7684\u4efb\u52a1\u5c31\u662f",
    "\u8fd9\u8282\u8bfe\u7684\u4efb\u52a1\u5c31\u662f",
    "\u4eca\u5929\u7684\u4efb\u52a1\u5c31\u662f",
    "\u63a5\u4e0b\u6765\u6211\u4eec\u8981\u89e3\u51b3\u7684\u662f",
    "\u6211\u4eec\u8981\u89e3\u51b3\u7684\u662f",
)
ADDRESSEE_FILLER_WORDS = ("\u554a", "\u5440", "\u5462", "\u5443", "\u989d", "\u54e6", "\u8bf6", "\u6b38")
QUESTION_REFERENCE_PATTERNS = (
    "\u4f60\u7684\u95ee\u9898",
    "\u8fd9\u4e2a\u95ee\u9898",
    "\u8fd9\u9053\u95ee\u9898",
    "\u5e26\u7740\u4f60\u7684\u95ee\u9898",
    "\u5e26\u7740\u8fd9\u4e2a\u95ee\u9898",
    "\u95ee\u9898\u7ee7\u7eed",
)
WEAK_QUESTION_HINT_WORDS = ("\u5462",)
SPOKEN_FILLER_WORDS = ("\u5443", "\u989d", "\u554a", "\u54e6", "\u5462")


class ActionTargetPipeline:
    def __init__(
        self,
        intent_model_path: str | Path,
        target_model_path: str | Path,
        people_config_path: str | Path,
        target_threshold: float = 0.5,
        intent_margin_threshold: float = 0.2,
    ) -> None:
        self.intent_model = joblib.load(intent_model_path)
        self.target_extractor = TargetExtractor(
            model_path=target_model_path,
            people_config_path=people_config_path,
            threshold=target_threshold,
        )
        self.intent_margin_threshold = intent_margin_threshold

    def _is_teaching_leadin(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized.startswith(TEACHING_LEADIN_PREFIXES):
            return False
        if "\uff1f" in normalized or "?" in normalized:
            return False
        return True

    def _has_question_hint(self, text: str) -> bool:
        normalized = text.strip()
        matched_words = [word for word in QUESTION_HINT_WORDS if word in normalized]
        if not matched_words:
            return False
        non_object_words = [
            word for word in matched_words if word not in ("\u95ee\u9898", *WEAK_QUESTION_HINT_WORDS)
        ]
        if non_object_words:
            return True
        if any(word in matched_words for word in WEAK_QUESTION_HINT_WORDS):
            return False
        return not any(pattern in normalized for pattern in QUESTION_REFERENCE_PATTERNS)

    def _is_teaching_topic_fragment(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if "\uff1f" in normalized or "?" in normalized:
            return False
        if self._has_question_hint(normalized):
            return False
        if any(word in normalized for word in ANSWER_HINT_WORDS):
            return False
        if self._is_second_person_question(normalized) or self._is_request_question(normalized):
            return False
        return any(word in normalized for word in TEACHING_TOPIC_HINT_WORDS)

    def _is_teaching_objective_statement(self, text: str) -> bool:
        normalized = text.strip()
        compact = normalized
        for word in SPOKEN_FILLER_WORDS:
            compact = compact.replace(word, "")
        compact = compact.rstrip(" \t\r\n\uff0c,\u3002\uff1b;")
        if not normalized:
            return False
        if "\uff1f" in normalized or "?" in normalized:
            return False
        if self._is_second_person_question(normalized) or self._is_request_question(normalized):
            return False
        if not any(pattern in compact for pattern in TEACHING_OBJECTIVE_PREFIX_PATTERNS):
            return False
        return any(word in compact for word in TEACHING_TOPIC_HINT_WORDS) or compact.endswith("\u600e\u4e48")

    def _is_target_direct_addressee(self, text: str, target: str) -> bool:
        index = text.find(target)
        if index == -1:
            return False
        prefix = text[max(0, index - 6):index].replace(" ", "")
        suffix = text[index + len(target): index + len(target) + 10]
        normalized_suffix = suffix.replace(" ", "")
        if (
            index <= 6
            or any(prefix.endswith(marker) for marker in ("请问", "那么", "那", "呃", "额", "好", "好的"))
        ) and (
            normalized_suffix.startswith(("同学", "，同学", ",同学"))
            or normalized_suffix.startswith(("\u4f60", "\uff0c\u4f60", ",\u4f60"))
        ):
            return True
        if normalized_suffix.startswith("\u540c\u5b66"):
            normalized_suffix = normalized_suffix[2:]
            normalized_suffix = normalized_suffix.lstrip("\uff0c,\u3002")
            while normalized_suffix.startswith(ADDRESSEE_FILLER_WORDS):
                normalized_suffix = normalized_suffix[1:]
            normalized_suffix = normalized_suffix.lstrip("\uff0c,\u3002")
        return normalized_suffix.startswith(("\u4f60", "\uff0c\u4f60", ",\u4f60", "\u4f60\u4eec", "\uff0c\u4f60\u4eec", ",\u4f60\u4eec"))

    def _is_start_class_statement(self, text: str) -> bool:
        normalized = text.strip()
        return any(word in normalized for word in START_CLASS_HINT_WORDS)

    def _is_end_class_command(self, text: str) -> bool:
        normalized = text.strip()
        if any(pattern in normalized for pattern in END_CLASS_FALSE_POSITIVE_PATTERNS):
            return False
        return any(word in normalized for word in END_CLASS_HINT_WORDS)

    def _is_second_person_question(self, text: str) -> bool:
        normalized = text.strip()
        has_question_hint = self._has_question_hint(normalized)
        return has_question_hint and any(pattern in normalized for pattern in SECOND_PERSON_QUESTION_PATTERNS)

    def _is_request_question(self, text: str) -> bool:
        normalized = text.strip()
        if normalized.startswith("\u6c42\u51fa") and ("\u5c31" in normalized or "\u4e86" in normalized):
            return False
        if any(pattern in normalized for pattern in REQUEST_QUESTION_PATTERNS):
            return True
        if normalized.startswith(REQUEST_QUESTION_PREFIXES):
            return True
        clauses = [part.strip() for part in re.split(r"[，,。；;：:]", normalized) if part.strip()]
        return any(
            clause.startswith(REQUEST_QUESTION_PREFIXES)
            and not (clause.startswith("\u6c42\u51fa") and ("\u5c31" in clause or "\u4e86" in clause))
            for clause in clauses
        )

    def _is_discovery_question(self, text: str) -> bool:
        normalized = text.strip()
        return any(pattern in normalized for pattern in DISCOVERY_QUESTION_PATTERNS)

    def _is_content_question(self, text: str) -> bool:
        normalized = text.strip()
        if any(pattern in normalized for pattern in CONTENT_REVIEW_PATTERNS):
            return False
        return any(pattern in normalized for pattern in CONTENT_QUESTION_PATTERNS)

    def _is_addressed_answer_prompt(self, text: str) -> bool:
        normalized = text.strip()
        return any(pattern in normalized for pattern in ADDRESSED_ANSWER_PROMPT_PATTERNS)

    def _is_addressed_question_prompt(self, text: str) -> bool:
        normalized = text.strip()
        return any(pattern in normalized for pattern in ADDRESSED_QUESTION_PROMPT_PATTERNS)

    def _is_plural_addressed_answer_prompt(self, text: str) -> bool:
        normalized = text.strip()
        return any(word in normalized for word in PLURAL_ADDRESSED_ANSWER_HINT_WORDS)

    def _references_target_as_content(self, text: str, target: str) -> bool:
        patterns = [f"{target}{suffix}" for suffix in REFERENCE_OBJECT_SUFFIXES]
        patterns.extend(
            [
                f"\u4f60\u660e\u767d{target}",
                f"\u4f60\u542c\u61c2{target}",
                f"\u4f60\u7406\u89e3{target}",
                f"\u5927\u5bb6\u89c9\u5f97{target}",
                f"\u4f60\u89c9\u5f97{target}",
                f"\u4f60\u8ba4\u4e3a{target}",
                f"{target}\u540c\u5b66\u7684",
                f"{target}\u7684\u610f\u601d\u662f",
                f"{target}\u610f\u601d\u662f",
                f"{target}\u8bf4\u7684\u662f",
                f"{target}\u8bb2\u7684\u662f",
            ]
        )
        return any(pattern in text for pattern in patterns)

    def _predict_action(self, text: str) -> str:
        action = self.intent_model.predict([text])[0]
        has_question_hint = self._has_question_hint(text)
        has_answer_hint = any(word in text for word in ANSWER_HINT_WORDS)
        has_discussion_hint = any(word in text for word in DISCUSSION_HINT_WORDS)
        has_vote_hint = any(word in text for word in VOTE_HINT_WORDS)
        has_start_class_hint = self._is_start_class_statement(text)
        has_end_class_hint = self._is_end_class_command(text)
        has_continue_hint = any(word in text for word in CONTINUE_HINT_WORDS)
        is_teaching_leadin = self._is_teaching_leadin(text)

        if has_discussion_hint:
            return "\u8ba8\u8bba"
        if has_vote_hint:
            return "\u4e3e\u624b\u8868\u51b3"
        if has_start_class_hint:
            return "\u7ee7\u7eed"
        if has_end_class_hint:
            return "\u4e0b\u8bfe"
        if self._is_request_question(text):
            return "\u63d0\u95ee"
        if action == "\u63d0\u95ee" and (
            (not has_question_hint and has_continue_hint)
            or is_teaching_leadin
            or self._is_teaching_topic_fragment(text)
            or self._is_teaching_objective_statement(text)
        ):
            return "\u7ee7\u7eed"

        if not hasattr(self.intent_model, "decision_function"):
            return action

        scores = self.intent_model.decision_function([text])
        if scores.ndim == 1:
            return action

        row = np.asarray(scores[0], dtype=float)
        if row.size < 2:
            return action

        top_two = np.sort(row)[-2:]
        margin = float(top_two[-1] - top_two[-2])
        has_clear_intent_hint = (
            has_question_hint
            or has_answer_hint
            or has_discussion_hint
            or has_vote_hint
            or has_start_class_hint
            or has_end_class_hint
            or self._is_request_question(text)
        )
        if margin < self.intent_margin_threshold and not has_clear_intent_hint:
            return "\u7ee7\u7eed"
        return action

    def _is_volunteer_question(self, text: str) -> bool:
        return any(word in text for word in VOLUNTEER_QUESTION_HINT_WORDS)

    def _is_feedback_statement(self, text: str) -> bool:
        if any(word in text for word in QUESTION_HINT_WORDS):
            return False
        return any(word in text for word in FEEDBACK_CONTINUE_HINT_WORDS)

    def predict(self, text: str) -> dict[str, object]:
        action = self._predict_action(text)
        target_result = self.target_extractor.predict(text)
        targets = target_result["targets"]
        has_question_hint = self._has_question_hint(text)

        if self._is_feedback_statement(text):
            action = "\u7ee7\u7eed"
            targets = []

        if self._is_volunteer_question(text):
            action = "\u63d0\u95ee"
            targets = []

        if action == "\u4e0b\u8bfe":
            targets = []

        if action == "\u7ee7\u7eed" and not targets and (
            self._is_second_person_question(text)
            or self._is_request_question(text)
            or self._is_discovery_question(text)
            or self._is_content_question(text)
        ):
            action = "\u63d0\u95ee"

        if self._is_addressed_question_prompt(text):
            if len(targets) >= 2 or self._is_plural_addressed_answer_prompt(text):
                action = "\u8ba8\u8bba"
                targets = targets[: self.target_extractor.max_targets]
            else:
                action = "\u63d0\u95ee"
                targets = []

        if targets and self._is_addressed_answer_prompt(text):
            action = "\u56de\u7b54"

        if action == "\u56de\u7b54" and not targets:
            action = "\u63d0\u95ee"
        elif action == "\u56de\u7b54":
            first_target = targets[0]
            if self._references_target_as_content(text, first_target):
                action = "\u63d0\u95ee" if has_question_hint else "\u7ee7\u7eed"
                targets = []
            else:
                if self._is_plural_addressed_answer_prompt(text):
                    targets = targets[: self.target_extractor.max_targets]
                else:
                    targets = targets[:1]

        # Normalize question/answer output:
        # - question-like utterances are resolved within question/answer
        # - question must not carry targets unless it directly calls on someone to answer
        # - answer must carry a target
        # - mixed self-question/explanation utterances stay as question
        if has_question_hint:
            if self._is_teaching_objective_statement(text):
                action = "\u7ee7\u7eed"
                targets = []
            elif targets and (
                self._is_second_person_question(text)
                or self._is_request_question(text)
                or self._is_addressed_answer_prompt(text)
                or (action == "\u56de\u7b54" and self._is_target_direct_addressee(text, targets[0]))
            ):
                first_target = targets[0]
                if self._references_target_as_content(text, first_target):
                    action = "\u63d0\u95ee"
                    targets = []
                else:
                    action = "\u56de\u7b54"
                    if self._is_plural_addressed_answer_prompt(text):
                        targets = targets[: self.target_extractor.max_targets]
                    else:
                        targets = targets[:1]
            else:
                action = "\u63d0\u95ee"
                targets = []
        elif action == "\u63d0\u95ee":
            targets = []
        elif action == "\u56de\u7b54" and not targets:
            action = "\u63d0\u95ee"

        return {
            "action": action,
            "target": targets,
        }
