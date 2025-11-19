"""CSV builder utilities for analysis outputs.

analysis/{game_id}/json 以下に溜めた A/B/C の結果から、
以下の2種類の CSV を生成する。

1. talk 単位 CSV
   path: analysis/{game_id}/csv/{analyzer_name}/talk/{analyzer_name}.csv
   columns:
       - text                (talk内容)
       - mentions            (JSON文字列: list[{"target", "valence", "raw}] or null)
       - affect              ("P1"〜"P5")
       - commitment_label    (C のラベル名 or 空)
       - weight              (C の weight or 空)

2. agent 単位 CSV
   path: analysis/{game_id}/csv/{analyzer_name}/agent/{analyzer_name}.csv
   columns:
       - agent_name
       - co_roles            (JSON文字列: list[str])
       - suspicion           (S1〜S5)
       - commitment_labels   (JSON文字列: list[{"commitment_label", "target_agent", "weight"}])
       - obligation_label    (直近の obligation ラベル or 空)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .analysis_paths import (
    get_csv_root,
    talk_analysis_json_path,
    obligation_store_json_path,
    commitment_store_json_path,
)
from .suspicion import valence_to_suspicion_label

if TYPE_CHECKING:
    from agent.agent import Agent


# ---- 共通ヘルパ -----------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONLines ファイルを読み込んで dict のリストとして返す."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 壊れた行はスキップ
                continue
    return rows


# ---- 1. talk CSV ---------------------------------------------------------


def build_talk_csv_for_analyzer(game_id: str, analyzer_name: str) -> None:
    """talk_analysis + commitment_store から talk CSV を生成する.

    出力:
        analysis/{game_id}/csv/{analyzer_name}/talk/{analyzer_name}.csv
    """
    # A: talk_analysis
    talk_path = talk_analysis_json_path(game_id, analyzer_name)
    talk_records = _load_jsonl(talk_path)
    if not talk_records:
        # まだ何も解析していない場合は何もしない
        return

    # C: commitment_store/{analyzer}/{speaker}.json を全部読み込んで index を貼る
    commitment_root = commitment_store_json_path(game_id, analyzer_name, "__DUMMY__").parent
    commitments_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    if commitment_root.exists():
        for speaker_file in commitment_root.glob("*.json"):
            speaker_records = _load_jsonl(speaker_file)
            for rec in speaker_records:
                speaker = rec.get("speaker")
                text = rec.get("text")
                if not speaker or text is None:
                    continue
                # 同じ (speaker, text) が複数ある場合は最後のものを優先
                commitments_by_key[(speaker, text)] = rec

    # 出力先ディレクトリ
    base_dir = get_csv_root(game_id) / analyzer_name / "talk"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"{analyzer_name}.csv"

    fieldnames = ["text", "mentions", "affect", "commitment_label", "weight"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in talk_records:
            text = rec.get("text", "")
            speaker = rec.get("speaker", "")
            mentions = rec.get("mentions")
            affect = rec.get("affect")

            comm = commitments_by_key.get((speaker, text), {})
            c_label = comm.get("commitment_label")
            c_weight = comm.get("weight")

            writer.writerow(
                {
                    "text": text,
                    "mentions": json.dumps(mentions, ensure_ascii=False),
                    "affect": affect,
                    "commitment_label": c_label or "",
                    "weight": c_weight or "",
                },
            )


# ---- 2. agent CSV --------------------------------------------------------


def _compute_co_roles_for_agent(
    talk_records: List[Dict[str, Any]],
    target_agent_name: str,
) -> List[str]:
    """target_agent が CO した役職の履歴（順序付き）を抽出する."""
    roles: list[str] = []
    for rec in talk_records:
        if rec.get("speaker") != target_agent_name:
            continue
        co_roles = rec.get("co_roles") or []
        for role in co_roles:
            roles.append(str(role))
    return roles


def _compute_suspicion_for_agent(
    talk_records: List[Dict[str, Any]],
    target_agent_name: str,
) -> str:
    """mentions から target_agent に対する suspicion ラベルを計算する."""
    vals: list[float] = []
    for rec in talk_records:
        for m in rec.get("mentions") or []:
            if m.get("target") == target_agent_name:
                try:
                    vals.append(float(m.get("valence", 0.0)))
                except (TypeError, ValueError):
                    continue
    return valence_to_suspicion_label(vals)


def _compute_commitment_labels_for_agent(
    game_id: str,
    analyzer_name: str,
    target_agent_name: str,
) -> List[Dict[str, Any]]:
    """commitment_store から target_agent が発話者のコミットメント一覧を取得する.

    形式:
        [
            {
                "commitment_label": str,
                "target_agent": str | None,
                "weight": "heavy" | "medium" | "light" | None
            },
            ...
        ]
    """
    path = commitment_store_json_path(game_id, analyzer_name, target_agent_name)
    rows = _load_jsonl(path)
    if not rows:
        return []

    commitments: list[dict[str, Any]] = []
    for rec in rows:
        label = rec.get("commitment_label")
        if not label:
            # commitment_label が無い行はスキップ
            continue
        commitments.append(
            {
                "commitment_label": label,
                "target_agent": rec.get("target_agent"),
                "weight": rec.get("weight"),
            },
        )

    return commitments


def _compute_obligation_label_for_agent(
    game_id: str,
    analyzer_name: str,
    target_agent_name: str,
) -> str:
    """obligation_store から target_agent への直近 obligation ラベルを取得する."""
    path = obligation_store_json_path(game_id, analyzer_name, target_agent_name)
    rows = _load_jsonl(path)
    if not rows:
        return ""
    # 最後の1件を「最新」として採用
    last = rows[-1]
    return str(last.get("obligation") or "")


def build_agent_csv_for_analyzer(
    game_id: str,
    analyzer_name: str,
    all_agent_names: List[str],
) -> None:
    """talk_analysis + commitment_store + obligation_store から agent CSV を生成する.

    出力:
        analysis/{game_id}/csv/{analyzer_name}/agent/{analyzer_name}.csv

    all_agent_names:
        suspicion や co_roles 等の対象となる全エージェント名のリスト
        （通常は info.status_map.keys() などを渡す）
    """
    talk_path = talk_analysis_json_path(game_id, analyzer_name)
    talk_records = _load_jsonl(talk_path)
    if not talk_records:
        # まだ何も解析していない場合は何もしない
        return

    base_dir = get_csv_root(game_id) / analyzer_name / "agent"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"{analyzer_name}.csv"

    fieldnames = [
        "agent_name",
        "co_roles",
        "suspicion",
        "commitment_labels",
        "obligation_label",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for target in all_agent_names:
            co_roles = _compute_co_roles_for_agent(talk_records, target)
            suspicion = _compute_suspicion_for_agent(talk_records, target)
            commitment_labels = _compute_commitment_labels_for_agent(
                game_id,
                analyzer_name,
                target,
            )
            obligation_label = _compute_obligation_label_for_agent(
                game_id,
                analyzer_name,
                target,
            )

            writer.writerow(
                {
                    "agent_name": target,
                    "co_roles": json.dumps(co_roles, ensure_ascii=False),
                    "suspicion": suspicion,
                    "commitment_labels": json.dumps(
                        commitment_labels,
                        ensure_ascii=False,
                    ),
                    "obligation_label": obligation_label,
                },
            )


# ---- Agent から呼ぶためのラッパ -----------------------------------------


def rebuild_csv_for_agent(analyzer: Agent) -> None:
    """指定エージェント視点の talk/agent CSV を両方再生成する.

    - analyzer: この CSV を生成するエージェントインスタンス
    """
    game_id = analyzer.game_id
    analyzer_name = analyzer.agent_name

    # talk CSV
    build_talk_csv_for_analyzer(game_id, analyzer_name)

    # agent CSV
    # ※ suspicion 等は全エージェント名が必要なので info から取る
    if analyzer.info is not None:
        all_agent_names = list(analyzer.info.status_map.keys())
    else:
        # info 未設定の場合は、とりあえず自分だけ
        all_agent_names = [analyzer_name]

    build_agent_csv_for_analyzer(game_id, analyzer_name, all_agent_names)
