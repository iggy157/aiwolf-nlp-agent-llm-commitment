from pathlib import Path

def get_analysis_root(game_id: str) -> Path:
    return Path("analysis") / game_id

def get_json_root(game_id: str) -> Path:
    return get_analysis_root(game_id) / "json"

def get_csv_root(game_id: str) -> Path:
    return get_analysis_root(game_id) / "csv"

# --- JSON ---

def talk_analysis_json_path(game_id: str, agent_name: str) -> Path:
    """各エージェントごとの A 分析結果 (talk_analysis/{agent}.json)."""
    return get_json_root(game_id) / "talk_analysis" / f"{agent_name}.json"

def obligation_store_json_path(game_id: str, analyzer_name: str, target_agent_name: str) -> Path:
    """B の obligation を保存するパス.

    以前: obligation_store/{target}.json （解析エージェントに依存せず）
    修正後: obligation_store/{analyzer}/{target}.json

    これにより、各エージェントごとに独立した obligation ストアを持てる。
    """
    return (
        get_json_root(game_id)
        / "obligation_store"
        / analyzer_name
        / f"{target_agent_name}.json"
    )

def commitment_store_json_path(game_id: str, agent_name: str) -> Path:
    """C の commitment を保存するパス.

    コミットメントは話者ごとにまとめたい想定なので、
    ここは従来どおり speaker 単位で 1 ファイル.
    """
    return get_json_root(game_id) / "commitment_store" / f"{agent_name}.json"

# --- CSV ---

def talk_csv_path(game_id: str, agent_name: str) -> Path:
    return get_csv_root(game_id) / "talk" / f"{agent_name}.csv"

def agent_csv_path(game_id: str, agent_name: str) -> Path:
    return get_csv_root(game_id) / "agent" / f"{agent_name}.csv"
