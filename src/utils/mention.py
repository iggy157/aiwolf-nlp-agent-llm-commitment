# src/utils/mention.py
"""Mention analysis utilities.

発話内の言及対象と態度（valence）を抽出するモジュール.
感情極性は外部の大規模辞書（sentiment_lexicon）を用いて算出する。
"""

from __future__ import annotations

from typing import Any, List

from aiwolf_nlp_common.packet import Talk
from .sentiment_lexicon import sentiment_score_for_text


def _normalize_text(text: str) -> str:
    """非常に簡易な正規化（必要に応じて拡張する）."""
    return text.strip()


def _window(text: str, center: int, radius: int = 15) -> str:
    """text の center 付近 radius 文字だけを切り出す."""
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    return text[start:end]


def extract_mentions(talk: Talk, alive_agents: List[str]) -> List[dict[str, Any]]:
    """発話内の言及対象と態度（valence）を抽出する.

    Args:
        talk: 1行分の Talk オブジェクト
        alive_agents: 現在生存しているエージェント名のリスト

    Returns:
        list[dict[str, Any]]:
            [
              {
                "target": "kanolab2",
                "valence": -0.7,
                "raw": "kanolab2は怪しいと思います",
              },
              ...
            ]

        言及がない場合は空リスト [] を返す。
    """
    text = _normalize_text(talk.text)
    if not text:
        return []

    mentions: list[dict[str, Any]] = []
    lower_text = text.lower()

    for agent_name in alive_agents:
        lower_agent = agent_name.lower()

        found_any = False

        # 1) 生文字のエージェント名
        start_idx = 0
        while True:
            idx = lower_text.find(lower_agent, start_idx)
            if idx == -1:
                break
            found_any = True
            snippet = _window(text, idx)
            valence = sentiment_score_for_text(snippet)
            mentions.append(
                {
                    "target": agent_name,
                    "valence": valence,
                    "raw": snippet,
                },
            )
            start_idx = idx + len(lower_agent)

        # 2) "@名前" 形式
        at_pattern = f"@{lower_agent}"
        start_idx = 0
        while True:
            idx = lower_text.find(at_pattern, start_idx)
            if idx == -1:
                break
            found_any = True
            snippet = _window(text, idx)
            valence = sentiment_score_for_text(snippet)
            mentions.append(
                {
                    "target": agent_name,
                    "valence": valence,
                    "raw": snippet,
                },
            )
            start_idx = idx + len(at_pattern)

        # どちらの形式でもヒットしていなければ何もしない
        if not found_any:
            continue

    return mentions
