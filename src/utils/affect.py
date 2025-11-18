# src/utils/affect.py
"""Affect analysis utilities.

発話全体の感情極性を、外部の感情辞書に基づいて5段階ラベル化する。
"""

from __future__ import annotations

from aiwolf_nlp_common.packet import Talk
from .sentiment_lexicon import sentiment_score_for_text


def _score_to_label(score: float) -> str:
    """スコア(-1.0～+1.0)を5段階ラベル(P1～P5)に変換する."""
    # ざっくりな分割例:
    #   -1.0 ～ -0.6 -> P1 (かなりネガ)
    #   -0.6 ～ -0.2 -> P2 (ややネガ)
    #   -0.2 ～  0.2 -> P3 (中立)
    #    0.2 ～  0.6 -> P4 (ややポジ)
    #    0.6 ～  1.0 -> P5 (かなりポジ)
    if score <= -0.6:
        return "P1"
    if score <= -0.2:
        return "P2"
    if score < 0.2:
        return "P3"
    if score < 0.6:
        return "P4"
    return "P5"


def analyze_affect(talk: Talk) -> str:
    """発話全体の感情極性ラベル（5段階）を返す.

    Args:
        talk: 1行分の Talk オブジェクト

    Returns:
        str: "P1" ～ "P5" のラベル
    """
    text = (talk.text or "").strip()
    if not text:
        return "P3"  # 中立扱い

    score = sentiment_score_for_text(text)
    return _score_to_label(score)
