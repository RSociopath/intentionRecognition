from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from target_extractor import TargetExtractor


QUESTION_HINT_WORDS = (
    "提问",
    "问题",
    "疑问",
    "举手",
    "多少",
    "几",
    "吗",
    "呢",
    "为什么",
    "怎么",
    "怎样",
    "是否",
    "？",
    "?",
)
ANSWER_HINT_WORDS = ("回答", "说说", "解释", "讲讲", "补充", "作答", "解答", "说明")
DISCUSSION_HINT_WORDS = ("讨论", "交流", "商量")
VOTE_HINT_WORDS = ("表决", "投票", "赞成", "反对")
END_CLASS_HINT_WORDS = (
    "下课",
    "这节课到这里",
    "今天这节课先上到这里",
    "今天就讲到这里，下课",
    "可以下课了",
    "同学们下课",
    "这节课结束",
    "准备下课",
)
END_CLASS_FALSE_POSITIVE_PATTERNS = (
    "下课本",
    "看下课本",
    "翻下课本",
)
VOLUNTEER_QUESTION_HINT_WORDS = ("谁能", "谁来", "谁可以", "谁愿意", "哪位同学", "哪个同学", "有没有人")
REFERENCE_OBJECT_SUFFIXES = ("的问题", "的疑问", "的困惑", "的问题点", "的想法", "的思路", "的做法")
FEEDBACK_CONTINUE_HINT_WORDS = (
    "答对了",
    "答错了",
    "说得对",
    "说得很好",
    "回答得很好",
    "回答得不错",
    "做得好",
    "做得不错",
    "不错",
    "很好",
)
CONTINUE_HINT_WORDS = (
    "公式是",
    "即",
    "等于",
    "表示",
    "我们已经知道",
    "定义",
    "概念",
    "性质",
    "我们发现",
    "那么",
    "就是",
    "也就是",
    "有关",
    "相关",
    "平方有关",
    "区别在于",
    "意味着",
    "说明",
)
TEACHING_LEADIN_PREFIXES = (
    "我们来",
    "接下来我们",
    "下面我们",
    "现在我们",
    "这一题我们",
    "这道题我们",
)


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
        if "？" in normalized or "?" in normalized:
            return False
        return True

    def _predict_action(self, text: str) -> str:
        action = self.intent_model.predict([text])[0]
        has_question_hint = any(word in text for word in QUESTION_HINT_WORDS)
        has_answer_hint = any(word in text for word in ANSWER_HINT_WORDS)
        has_discussion_hint = any(word in text for word in DISCUSSION_HINT_WORDS)
        has_vote_hint = any(word in text for word in VOTE_HINT_WORDS)
        has_end_class_hint = self._is_end_class_command(text)
        has_continue_hint = any(word in text for word in CONTINUE_HINT_WORDS)
        is_teaching_leadin = self._is_teaching_leadin(text)

        if has_discussion_hint:
            return "讨论"
        if has_vote_hint:
            return "举手表决"
        if has_end_class_hint:
            return "下课"
        if action == "提问" and ((not has_question_hint and has_continue_hint) or is_teaching_leadin):
            return "继续"

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
        has_clear_intent_hint = has_question_hint or has_answer_hint or has_discussion_hint or has_vote_hint or has_end_class_hint
        if margin < self.intent_margin_threshold and not has_clear_intent_hint:
            return "继续"
        return action

    def _is_end_class_command(self, text: str) -> bool:
        normalized = text.strip()
        if any(pattern in normalized for pattern in END_CLASS_FALSE_POSITIVE_PATTERNS):
            return False
        return any(word in normalized for word in END_CLASS_HINT_WORDS)

    def _is_volunteer_question(self, text: str) -> bool:
        return any(word in text for word in VOLUNTEER_QUESTION_HINT_WORDS)

    def _is_feedback_statement(self, text: str) -> bool:
        return any(word in text for word in FEEDBACK_CONTINUE_HINT_WORDS)

    def predict(self, text: str) -> dict[str, object]:
        action = self._predict_action(text)
        target_result = self.target_extractor.predict(text)
        targets = target_result["targets"]

        if self._is_feedback_statement(text):
            action = "继续"
            targets = []

        if self._is_volunteer_question(text):
            action = "提问"
            targets = []

        if action == "下课":
            targets = []

        if action == "回答" and not targets:
            action = "提问"
        elif action == "回答":
            first_target = targets[0]
            if any(f"{first_target}{suffix}" in text for suffix in REFERENCE_OBJECT_SUFFIXES):
                action = "提问"
                targets = []
            else:
                targets = targets[:1]

        return {
            "action": action,
            "target": targets,
        }
