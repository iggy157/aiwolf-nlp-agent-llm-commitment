# utils/co.py
"""CO（カミングアウト）役職の抽出ユーティリティ.

発話テキストから「自分がどの役職かをCOしているか」を判定する。
日本語および英語の代表的な表現に対応する。
"""

from __future__ import annotations

import re
from typing import List

from aiwolf_nlp_common.packet import Talk


# 役職ごとの CO パターン
# - pronoun/self: 私は, 自分は, 俺は, 僕は, I am, I'm など
# - CO: 「CO」「ＣＯ」「co」など
# - 役職名: 日本語 + 英語
SELF_PAT = r"(?:私は|自分は|俺は|僕は|わたしは|おれは|ぼくは|I am|I'm|im|I\'m)"
CO_PAT = r"(?:CO|ＣＯ|Co|co)"
SPACE_OPT = r"[ 　]*"

ROLE_PATTERNS = {
    "seer": [
        # CO pattern
        rf"{CO_PAT}{SPACE_OPT}(?:占い師|占い|seers?|fortune ?teller)s?",
        # Self-introduction pattern
        rf"{SELF_PAT}{SPACE_OPT}(?:占い師|占い|seer|fortune ?teller)(?:です|だ|だよ|です。|だ。)?",
        # English only
        r"(?:I am|I'm|Im)\s+the\s+seer",
    ],
    "medium": [
        rf"{CO_PAT}{SPACE_OPT}(?:霊能者?|霊媒師?|medium)",
        rf"{SELF_PAT}{SPACE_OPT}(?:霊能者?|霊媒師?|medium)(?:です|だ|だよ|です。|だ。)?",
        r"(?:I am|I'm|Im)\s+the\s+medium",
    ],
    "bodyguard": [
        rf"{CO_PAT}{SPACE_OPT}(?:狩人|ボディガード|護衛|bodyguard|guard)",
        rf"{SELF_PAT}{SPACE_OPT}(?:狩人|ボディガード|護衛|bodyguard|guard)(?:です|だ|だよ|です。|だ。)?",
        r"(?:I am|I'm|Im)\s+the\s+bodyguard",
        r"(?:I am|I'm|Im)\s+the\s+guard",
    ],
    "villager": [
        rf"{CO_PAT}{SPACE_OPT}(?:村人|市民|villager|villager?s?)",
        rf"{SELF_PAT}{SPACE_OPT}(?:村人|市民|villager)(?:です|だ|だよ|です。|だ。)?",
        r"(?:I am|I'm|Im)\s+villager",
        r"(?:I am|I'm|Im)\s+a\s+villager",
    ],
    "werewolf": [
        rf"{CO_PAT}{SPACE_OPT}(?:人狼|狼|werewolf|wolf)",
        rf"{SELF_PAT}{SPACE_OPT}(?:人狼|狼|werewolf|wolf)(?:です|だ|だよ|です。|だ。)?",
        r"(?:I am|I'm|Im)\s+a?\s*werewolf",
        r"(?:I am|I'm|Im)\s+a?\s*wolf",
    ],
    "possessed": [
        rf"{CO_PAT}{SPACE_OPT}(?:狂人|狂信者|possessed)",
        rf"{SELF_PAT}{SPACE_OPT}(?:狂人|狂信者|possessed)(?:です|だ|だよ|です。|だ。)?",
        r"(?:I am|I'm|Im)\s+the\s+possessed",
    ],
}


# コンパイル済みパターン（role -> list[Pattern]）
COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    role: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for role, patterns in ROLE_PATTERNS.items()
}


def extract_co_roles(talk: Talk) -> List[str]:
    """当該発話でのCO役職を抽出する.

    - 戻り値は役職名のリスト（["seer"], ["villager", "seer"] など）
    - 該当する CO がない場合は空リスト [] を返す。
    - 1 発話内に複数の役職パターンがヒットした場合、
      テキスト中の出現位置順にソートして返す。
    """
    text = talk.text
    if not text:
        return []

    matches: list[tuple[int, str]] = []  # (start_index, role)

    for role, patterns in COMPILED_PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(text):
                matches.append((m.start(), role))

    if not matches:
        return []

    # 出現位置順にソート
    matches.sort(key=lambda x: x[0])

    # 役職名だけを順番に返す（重複は一応許容、必要ならset化）
    return [role for _, role in matches]
