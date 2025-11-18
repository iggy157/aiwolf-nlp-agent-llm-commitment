# src/utils/analysis_talk.py
"""Talk-level analysis (A + B + C).

talk history から「1 発話ごとの分析結果(A)」を生成し、
さらに B（obligation）、C（commitment）も同時に処理して保存するモジュール。

A の出力（talk_analysis）:
- text:      該当発話内容 (str)
- speaker:   該当発話の発話者 (str)
- mentions:  発話内の言及対象と態度（valence）
             list[{"target": str, "valence": float, "raw": str}] or null
- co_roles:  当該発話でのCO役職 list[str] or null
- affect:    発話全体の感情極性ラベル（"P1"〜"P5"）

B の出力（obligation_store/{target}.json, JSONLines）:
- from_agent:    発話者
- target_agent:  義務を負うターゲット
- text:          該当発話内容
- obligation:    "answer_obligation" / "defence_obligation" / "acknowledge_obligation"

C の出力（commitment_store/{speaker}.json, JSONLines）:
- speaker:           発話者
- text:              該当発話内容
- commitment_label:  コミットメントラベル名 or null
- target_agent:      コミットメントの対象エージェント or null
- weight:            "heavy" / "medium" / "light" / null
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, List

from aiwolf_nlp_common.packet import Talk

from .analysis_paths import (
    talk_analysis_json_path,
    obligation_store_json_path,
)
from .mention import extract_mentions
from .co import extract_co_roles
from .affect import analyze_affect
from .obligation import derive_obligations_from_a_record
from .commitment import analyze_and_append_commitment

if TYPE_CHECKING:
    # 循環参照を避けるために型チェック時のみ import
    from agent.agent import Agent


def build_talk_analysis_record(
    talk: Talk,
    alive_agents: List[str],
) -> dict[str, Any]:
    """1 発話分の A 分析結果を作る。

    Args:
        talk: 解析対象の Talk 行
        alive_agents: 生存エージェント名のリスト

    Returns:
        dict[str, Any]: A の出力仕様に従ったレコード
    """
    text = talk.text or ""
    speaker = talk.agent

    # 言及と valence を抽出
    mentions = extract_mentions(talk, alive_agents)
    mentions_or_null: Any = mentions if mentions else None

    # CO 役職を抽出
    co_roles = extract_co_roles(talk)
    co_roles_or_null: Any = co_roles if co_roles else None

    # 発話全体の感情極性ラベル
    affect = analyze_affect(talk)  # "P1"〜"P5"

    record: dict[str, Any] = {
        "text": text,
        "speaker": speaker,
        "mentions": mentions_or_null,
        "co_roles": co_roles_or_null,
        "affect": affect,
    }
    return record


def append_talk_analysis_record(
    game_id: str,
    agent_name: str,
    record: dict[str, Any],
) -> None:
    """A の1レコードを talk_analysis/{agent}.json に追記（JSONLines）。"""
    path = talk_analysis_json_path(game_id, agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_obligations(
    game_id: str,
    obligations: List[dict[str, Any]],
) -> None:
    """B の obligation レコード群を obligation_store/{target}.json に追記。"""
    for ob in obligations:
        target = ob.get("target_agent")
        if not target:
            continue
        path = obligation_store_json_path(game_id, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ob, ensure_ascii=False) + "\n")


def run_talk_analysis_for_new_talks(
    agent: "Agent",
    start_index: int,
) -> int:
    """Agent の talk_history[start_index:] に対して A/B/C を実行し、結果を保存する。

    - A: talk_analysis/{self.agent_name}.json に JSONL で追記
    - B: obligation_store/{target}.json に JSONL で追記
    - C: commitment_store/{speaker}.json に JSONL で追記

    戻り値:
        新たに処理し終えた talk_history の末尾インデックス。
        Agent 側では last_analyzed_talk_index の更新に利用する。
    """
    talks: List[Talk] = agent.talk_history[start_index:]
    if not talks:
        return start_index

    game_id = agent.game_id
    agent_name = agent.agent_name
    alive_agents = agent.get_alive_agents()

    current_index = start_index

    for talk in talks:
        # --- A: talk_analysis ---
        a_record = build_talk_analysis_record(talk, alive_agents)
        append_talk_analysis_record(game_id, agent_name, a_record)

        # --- B: obligation_store ---
        obligations = derive_obligations_from_a_record(a_record)
        if obligations:
            append_obligations(game_id, obligations)

        # --- C: commitment_store ---
        mentions = a_record.get("mentions") or []
        analyze_and_append_commitment(
            agent=agent,
            talk=talk,
            mentions=mentions,
        )

        current_index += 1

    return current_index
