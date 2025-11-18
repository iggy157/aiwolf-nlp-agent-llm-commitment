"""Commitment analysis utilities (C).

llm1 を用いて、1 発話から commitment ラベルを抽出し、
analysis/{game_id}/json/commitment_store/{analyzer}/{speaker}.json に保存する。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List

from aiwolf_nlp_common.packet import Talk

from .analysis_paths import commitment_store_json_path

if TYPE_CHECKING:
    from agent.agent import Agent


def analyze_and_append_commitment(
    agent: "Agent",
    talk: Talk,
    mentions: List[dict[str, Any]] | None,
) -> None:
    """C: llm1 を使って commitment を付与し、commitment_store に保存する。

    Args:
        agent: 呼び出し元エージェント（llm1 呼び出しに使う & analyzer として記録）
        talk: 対象の Talk 行
        mentions: A で計算した mentions（None/[] も可）
    """
    # llm に渡す「言及先エージェント一覧」
    mentioned_agents = [m["target"] for m in (mentions or []) if "target" in m]

    extra_context = {
        "talk": talk,
        "mentioned_agents": mentioned_agents,
    }

    # llm1 + prompt "commitment" を使う
    response = agent._send_message_to_llm(
        request=None,
        llm_name="llm1",
        prompt_key_override="commitment",
        extra_context=extra_context,
        use_context_history=True,      # initialize_llm1 の文脈を使う
        add_to_context_history=False,  # 各発話ごとはコンテキストには積まない
    )
    if not response:
        return

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        agent.agent_logger.logger.warning(
            "Invalid commitment JSON for talk '%s': %s",
            talk.text,
            response,
        )
        return

    row = {
        "speaker": talk.agent,
        "text": talk.text,
        "commitment_label": data.get("commitment_label"),
        "target_agent": data.get("target_agent"),
        "weight": data.get("weight"),
    }

    # analyzer = この関数を呼び出した Agent（= agent.agent_name）
    analyzer_name = agent.agent_name
    speaker_name = talk.agent

    path = commitment_store_json_path(agent.game_id, analyzer_name, speaker_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
