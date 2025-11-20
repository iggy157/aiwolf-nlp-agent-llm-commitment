# src/utils/mention.py
"""Mention analysis utilities.

発話内の言及対象と態度（valence）を抽出するモジュール.
感情極性は外部の大規模辞書（sentiment_lexicon）を用いて算出する。

可能であれば spaCy を用いて文・トークン単位にフレーズを切り出し、
辞書ベースの感情スコアを各ターゲットに割り当てる。

spaCy / 日本語モデルが利用できない環境では、従来同様の
「@名前 周辺 window を切り出してスコアリングする」方式にフォールバックする。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aiwolf_nlp_common.packet import Talk
from .sentiment_lexicon import sentiment_score_for_text

# ------------------------------------------------------------
# spaCy ローダ（あれば使う、なければ None のまま）
# ------------------------------------------------------------
try:  # pragma: no cover - 外部依存のためテストしづらいので除外
    import spacy  # type: ignore

    _NLP: Optional["spacy.language.Language"]
    try:
        # GiNZA があるなら優先（日本語依存構造に強い）
        _NLP = spacy.load("ja_ginza")
    except Exception:  # noqa: BLE001
        try:
            # なければ汎用の日本語モデルを試す
            _NLP = spacy.load("ja_core_news_sm")
        except Exception:  # noqa: BLE001
            _NLP = None
except Exception:  # noqa: BLE001
    spacy = None  # type: ignore
    _NLP = None


def _normalize_text(text: str) -> str:
    """非常に簡易な正規化（必要に応じて拡張する）."""
    return (text or "").strip()


def _normalize_agent_name(name: str) -> str:
    """エージェント名の比較用正規化."""
    return name.strip().lstrip("@").lower()


# ============================================================
# spaCy ベース実装
# ============================================================


def _find_agent_tokens_in_sentence(
    sent,
    norm_agent_to_original: Dict[str, str],
) -> List[tuple]:
    """文中のトークンのうち、エージェント名に該当するものを列挙する.

    Returns:
        list of (token, original_agent_name)
    """
    results: List[tuple] = []
    for token in sent:
        surface = token.text.strip()
        norm_tok = surface.lstrip("@").lower()
        original = norm_agent_to_original.get(norm_tok)
        if original is not None:
            results.append((token, original))
    return results


def _span_for_agent_in_sentence(sent, agent_token, norm_agent_to_original: Dict[str, str]) -> str:
    """1文中で、あるエージェントに対する評価を含みそうなスパンを推定して返す.

    方針（完全ではないが、実用的なヒューリスティック）:
      - エージェントの直前 0〜3 トークンを左側に含める
        （例: 「冷静なリュウジ」などの形容も拾うため）
      - エージェントの位置から右側に向かって走査し、
        - 別のエージェント名が現れたところ
        - 文末記号（。！？!?）
        - 条件接続詞的な「けど」「が」（SCONJ/CCONJ）
        などを目安にフレーズを切り出す
    """
    doc = sent.doc
    sent_start = sent.start
    sent_end = sent.end

    # 左側: 直前の修飾語も少し含める（最大 3 トークン）
    start = agent_token.i
    for i in range(agent_token.i - 1, max(sent_start, agent_token.i - 3) - 1, -1):
        tok = doc[i]
        surface = tok.text.strip()
        norm = surface.lstrip("@").lower()
        if norm in norm_agent_to_original and tok.i != agent_token.i:
            break  # 他のエージェントを跨がない
        if surface in ("。", "！", "!", "？", "?"):
            break
        start = i

    # 右側: 別のエージェント・接続詞・文末記号などまで
    end = agent_token.i + 1
    for i in range(agent_token.i + 1, sent_end):
        tok = doc[i]
        surface = tok.text.strip()
        norm = surface.lstrip("@").lower()

        # 次のエージェント名が出たらここまで
        if norm in norm_agent_to_original:
            break

        # 文末記号でストップ
        if surface in ("。", "！", "!", "？", "?"):
            end = i
            break

        # 「けど」「が」などの接続で区切る（緩めのヒューリスティック）
        if surface in ("けど", "が") and tok.pos_ in ("SCONJ", "CCONJ"):
            end = i
            break

        end = i + 1

    if start >= end:
        # 念のため安全側に倒しておく
        start = agent_token.i
        end = min(agent_token.i + 1, sent_end)

    span = doc[start:end]
    return span.text


def _extract_mentions_spacy(talk: Talk, alive_agents: List[str]) -> List[dict[str, Any]]:
    """spaCy を用いた mention 抽出.

    - 入力: Talk + alive_agents
    - 出力: [{"target": agent_name, "valence": float, "raw": snippet}, ...]
    """
    assert _NLP is not None  # 呼び出し前にチェック済みのはず

    text = _normalize_text(talk.text)
    if not text:
        return []

    doc = _NLP(text)
    if not doc.sents:
        # 文分割できない場合は、そのまま全文を1文扱い
        sents = [doc[:]]
    else:
        sents = list(doc.sents)

    # エージェント名を正規化しておく
    norm_agent_to_original: Dict[str, str] = {}
    for name in alive_agents:
        norm = _normalize_agent_name(name)
        if not norm:
            continue
        # 同じ norm に複数名が来る可能性は低い想定なので上書きでOK
        norm_agent_to_original[norm] = name

    mentions: List[dict[str, Any]] = []

    for sent in sents:
        # この文に出てくるエージェントトークンを列挙
        agent_tokens = _find_agent_tokens_in_sentence(sent, norm_agent_to_original)
        if not agent_tokens:
            continue

        for agent_token, original_name in agent_tokens:
            snippet = _span_for_agent_in_sentence(sent, agent_token, norm_agent_to_original)
            valence = sentiment_score_for_text(snippet)
            mentions.append(
                {
                    "target": original_name,
                    "valence": valence,
                    "raw": snippet,
                }
            )

    return mentions


# ============================================================
# 従来方式（フォールバック用）
# ============================================================


def _window(text: str, center: int, radius: int = 15) -> str:
    """text の center 付近 radius 文字だけを切り出す."""
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    return text[start:end]


def _extract_mentions_window_based(talk: Talk, alive_agents: List[str]) -> List[dict[str, Any]]:
    """従来の「@名前 周辺 window + 辞書スコア」による mention 抽出."""
    text = _normalize_text(talk.text)
    if not text:
        return []

    mentions: List[dict[str, Any]] = []
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


# ============================================================
# 公開 API
# ============================================================


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

    実装詳細:
        - spaCy + 日本語モデルが利用可能な場合:
            文・トークン単位でフレーズを切り出し、
            sentiment_lexicon に基づいて各ターゲットの valence を計算する。
        - 利用できない場合:
            従来の「@名前 周辺 window を sentiment_score_for_text でスコア」
            する方式にフォールバックする。
    """
    if _NLP is not None:
        try:
            return _extract_mentions_spacy(talk, alive_agents)
        except Exception:
            # 何かあってもシステム全体が落ちないようにフォールバック
            return _extract_mentions_window_based(talk, alive_agents)
    else:
        return _extract_mentions_window_based(talk, alive_agents)
