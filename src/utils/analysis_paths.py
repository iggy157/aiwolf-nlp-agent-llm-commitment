# utils/analysis_paths.py
from pathlib import Path

def get_analysis_root(game_id: str) -> Path:
    return Path("analysis") / game_id

def get_json_root(game_id: str) -> Path:
    return get_analysis_root(game_id) / "json"

def get_csv_root(game_id: str) -> Path:
    return get_analysis_root(game_id) / "csv"

# --- JSON ---

def talk_analysis_json_path(game_id: str, agent_name: str) -> Path:
    return get_json_root(game_id) / "talk_analysis" / f"{agent_name}.json"

def obligation_store_json_path(game_id: str, agent_name: str) -> Path:
    return get_json_root(game_id) / "obligation_store" / f"{agent_name}.json"

def commitment_store_json_path(game_id: str, agent_name: str) -> Path:
    return get_json_root(game_id) / "commitment_store" / f"{agent_name}.json"

# --- CSV ---

def talk_csv_path(game_id: str, agent_name: str) -> Path:
    return get_csv_root(game_id) / "talk" / f"{agent_name}.csv"

def agent_csv_path(game_id: str, agent_name: str) -> Path:
    return get_csv_root(game_id) / "agent" / f"{agent_name}.csv"
