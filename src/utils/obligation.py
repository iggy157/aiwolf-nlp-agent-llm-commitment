# src/utils/obligation.py
"""Obligation analysis utilities (B).

A の出力（1発話レコード）から、obligation ラベルを導出するモジュール。

仕様:
- A の mentions が null / [] の場合は何もしない。
- 発話が疑問文なら:
    -> mention の target 全員に answer_obligation
- 発話が疑問文でない場合:
    -> mention.valence < 0 なら defence_obligation
    -> mention.valence > 0 なら acknowledge_obligation
"""

from __future__ import annotations

import re
from typing import Any, List, Dict

QUESTION_PAT = re.compile(r"(？|\?|ですか|ますか|だろうか|かな)")


def is_question_sentence(text: str) -> bool:
    """簡単な正規表現で、文が疑問文かどうかを判定."""
    return bool(QUESTION_PAT.search(text))


def derive_obligations_from_a_record(a_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A の1発話レコードから、0個以上の obligation レコードを導出する.

    Args:
        a_record: analysis_talk.build_talk_analysis_record の戻り値

    Returns:
        list[dict[str, Any]]:
            [
              {
                "from_agent": "kanolab1",
                "target_agent": "kanolab2",
                "text": "kanolab2は怪しいと思う",
                "obligation": "defence_obligation",
              },
              ...
            ]
    """
    text = a_record.get("text", "") or ""
    speaker = a_record.get("speaker")
    mentions = a_record.get("mentions") or []

    if not speaker or not mentions:
        return []

    obligations: list[dict[str, Any]] = []

    if is_question_sentence(text):
        # 質問文 → 全ターゲットに answer_obligation
        for m in mentions:
            target = m.get("target")
            if not target:
                continue
            obligations.append(
                {
                    "from_agent": speaker,
                    "target_agent": target,
                    "text": text,
                    "obligation": "answer_obligation",
                },
            )
    else:
        # 質問文でない場合、valence の符号を見る
        for m in mentions:
            target = m.get("target")
            if not target:
                continue
            valence = float(m.get("valence", 0.0))
            if valence < 0:
                label = "defence_obligation"
            elif valence > 0:
                label = "acknowledge_obligation"
            else:
                continue
            obligations.append(
                {
                    "from_agent": speaker,
                    "target_agent": target,
                    "text": text,
                    "obligation": label,
                },
            )

    return obligations
