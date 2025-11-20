# src/utils/sentiment_lexicon.py
"""Sentiment lexicon loader and scoring utilities.

日本語評価極性辞書（用言編 & 名詞編）から作った
data/sentiment_lexicon.tsv を読み込み、
単語→スコア(-1.0～+1.0) を提供するモジュール。

スコア計算は、辞書トークン長に基づく文字 n-gram の
最長一致セグメンテーションで行う。
"""

from __future__ import annotations

import csv
import re
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


@lru_cache(maxsize=1)
def _build_length_index() -> Dict[int, Dict[str, float]]:
    """token の長さ別インデックスを構築する.

    { 長さn: {token: score, ...}, ... } の構造にすることで、
    文字 n-gram による最長一致探索を効率化する。
    """
    lexicon = load_lexicon()
    length_index: Dict[int, Dict[str, float]] = {}
    for token, score in lexicon.items():
        n = len(token)
        if n <= 0:
            continue
        bucket = length_index.setdefault(n, {})
        bucket[token] = score
    return length_index


def _normalize_text_for_matching(text: str) -> str:
    """辞書マッチ用に簡易正規化する.

    - 前後空白を削除
    - 半角/全角スペースを削除（辞書トークンはスペースを含まない前提）
    将来的に、必要ならここに NFKC 等の正規化を追加してもよい。
    """
    s = text.strip()
    # 半角/全角スペース削除
    s = s.replace(" ", "").replace("　", "")
    # その他の制御文字的な空白も一応削っておく
    s = re.sub(r"\s+", "", s)
    return s


def sentiment_score_for_text(text: str) -> float:
    """テキスト全体の感情スコア(-1.0～+1.0)をざっくり計算する.

    辞書トークンの文字長に基づく最長一致セグメンテーションを行い、
    マッチしたトークンのスコアの平均を返す。

    具体的な手順:
        1. テキストを簡易正規化（空白類の除去など）
        2. 辞書トークンを長さ別にバケット化
        3. 文字列の左から右へ走査し、残り長さの範囲で
           「最も長いトークン長」から順に substring を辞書と照合
        4. マッチしたらそのトークンのスコアを加算し、インデックスをその分進める
        5. マッチしなければ1文字だけ進める
        6. 最終的に (総スコア / マッチ数) を [-1.0, 1.0] にクリップして返す

    Args:
        text: 対象テキスト（日本語）

    Returns:
        float: -1.0 ～ +1.0 のスコア
    """
    if not text:
        return 0.0

    lexicon = load_lexicon()
    if not lexicon:
        return 0.0

    length_index = _build_length_index()
    if not length_index:
        return 0.0

    norm_text = _normalize_text_for_matching(text)
    if not norm_text:
        return 0.0

    max_len = max(length_index.keys())
    n_chars = len(norm_text)

    hits = 0
    total = 0.0

    i = 0
    while i < n_chars:
        matched = False
        # 残り文字数の範囲で、長いトークンから順に試す
        max_token_len = min(max_len, n_chars - i)
        for L in range(max_token_len, 0, -1):
            bucket = length_index.get(L)
            if not bucket:
                continue
            sub = norm_text[i : i + L]
            score = bucket.get(sub)
            if score is not None:
                hits += 1
                total += score
                i += L
                matched = True
                break
        if not matched:
            i += 1

    if hits == 0:
        return 0.0

    avg = total / hits
    # [-1, 1] にクリップ
    if avg > 1.0:
        return 1.0
    if avg < -1.0:
        return -1.0
    return avg
