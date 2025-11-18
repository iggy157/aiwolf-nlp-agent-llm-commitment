# src/utils/suspicion.py
"""Suspicion label utilities.

mentions における valence の平均から、
エージェントごとの「疑われ度」ラベル (S1〜S5) を計算する。
"""

from __future__ import annotations

from typing import Sequence


def valence_to_suspicion_label(valences: Sequence[float]) -> str:
    """valence のリストから5段階の疑われ度ラベルを返す.

    仕様:
    - valence がマイナス寄りなほど「疑われている」とみなす。
    - 平均値が低いほど S5 に近づき、高いほど S1 に近づく。

    ざっくりな分割:
        avg <= -0.6 -> S5 (かなり怪しい)
        -0.6 < avg <= -0.2 -> S4 (やや怪しい)
        -0.2 < avg <  0.2 -> S3 (どちらとも言えない)
         0.2 <= avg < 0.6 -> S2 (やや白寄り)
         0.6 <= avg       -> S1 (かなり白寄り)
    """
    if not valences:
        # 情報がないときは中立
        return "S3"

    avg = sum(valences) / len(valences)

    if avg <= -0.6:
        return "S5"
    if avg <= -0.2:
        return "S4"
    if avg < 0.2:
        return "S3"
    if avg < 0.6:
        return "S2"
    return "S1"
