# src/utils/sentiment_lexicon.py
"""Sentiment lexicon loader and scoring utilities.

日本語評価極性辞書（用言編 & 名詞編）から作った
data/sentiment_lexicon.tsv を読み込み、
単語→スコア(-1.0～+1.0) を提供するモジュール。
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict


# プロジェクトルート/data/sentiment_lexicon.tsv を想定
DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve()
    .parents[2]  # .../project_root/
    / "data"
    / "sentiment_lexicon.tsv"
)


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> Dict[str, float]:
    """感情辞書を読み込んで token -> score の dict を返す.

    Args:
        path: 明示的に指定したい辞書ファイルパス（省略可）

    Returns:
        dict[str, float]: token -> score(-1.0～+1.0)
    """
    lexicon_path = Path(path) if path else DEFAULT_LEXICON_PATH
    lexicon: Dict[str, float] = {}

    if not lexicon_path.exists():
        # 辞書が無ければ空の辞書を返す（呼び出し側で0扱い）
        return lexicon

    with lexicon_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # 期待カラム: token, lang, score
        for row in reader:
            token = row.get("token")
            score_str = row.get("score")
            if not token or not score_str:
                continue
            try:
                score = float(score_str)
            except ValueError:
                continue
            lexicon[token] = score

    return lexicon


def sentiment_score_for_text(text: str) -> float:
    """テキスト全体の感情スコア(-1.0～+1.0)をざっくり計算する.

    現状は「辞書内単語の合計 / ヒット数」という簡易なもの。
    将来的に MeCab で形態素分割してから照合するように差し替え可能。

    Args:
        text: 対象テキスト（日本語）

    Returns:
        float: -1.0 ～ +1.0 のスコア
    """
    lexicon = load_lexicon()
    if not lexicon:
        return 0.0

    if not text:
        return 0.0

    hits = 0
    total = 0.0

    # ひとまず「そのまま部分一致」で見る簡易実装
    # 本気でやるなら MeCab などで分かち書き + 原形化してから token と照合する。
    for token, score in lexicon.items():
        if token in text:
            hits += 1
            total += score

    if hits == 0:
        return 0.0

    avg = total / hits
    # [-1, 1] にクリップ
    if avg > 1.0:
        return 1.0
    if avg < -1.0:
        return -1.0
    return avg
