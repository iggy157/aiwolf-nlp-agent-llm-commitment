"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread
from utils.analysis_talk import run_talk_analysis_for_new_talks
from utils.csv_builder import rebuild_csv_for_agent

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.game_id = game_id
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        # self.llm_model: BaseChatModel | None = None
        # self.llm_message_history: list[BaseMessage] = []
        self.llm_models: dict[str, BaseChatModel] = {}
        self.llm_message_histories: dict[str, list[BaseMessage]] = {}
        self.llm_context_histories: dict[str, list[BaseMessage]] = {}

        self.last_analyzed_talk_index: int = 0

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))
    
    def _create_llm_model(self, llm_type: str, cfg: dict[str, Any]) -> BaseChatModel:
        """LLMモデルインスタンスを生成するヘルパ.

        Args:
            llm_type (str): 'openai', 'google', 'ollama' など
            cfg (dict[str, Any]): model, temperature, base_url などを含む設定
                                    （llm.backends.<name> の中身、または旧来の openai/google セクションから組み立てたもの）
        """
        temperature = float(cfg.get("temperature", 0.7))
        model = str(cfg["model"])

        match llm_type:
            case "openai":
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=str(cfg.get("base_url", "http://localhost:11434")),
                )
            case _:
                raise ValueError(f"Unknown LLM type: {llm_type}")

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history = []
            self.whisper_history = []
            for history in self.llm_message_histories.values():
                history.clear()
            for history in self.llm_context_histories.values():
                history.clear()
            
            self.last_analyzed_talk_index = 0

        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _send_message_to_llm(
        self,
        request: Request | None,
        *,
        llm_name: str = "llm2",
        prompt_key_override: str | None = None,
        extra_context: dict[str, Any] | None = None,
        use_context_history: bool = False,     # コンテキスト用ノートを使うか
        add_to_context_history: bool = False,  # 今回のやりとりをコンテキストに足すか
    ) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): 処理するリクエストタイプ（INITIALIZE, TALK など）
            llm_name (str): 使用するLLMバックエンド名（llm1, llm2, default など）
            prompt_key_override (str | None): config["prompt"] のキーを明示指定する場合
            extra_context (dict[str, Any] | None): プロンプトに追加で埋め込むコンテキスト
            use_context_history (bool): True なら llm_context_histories をベースにする
            add_to_context_history (bool): True なら今回の Human/AI もコンテキストに積む
        """
        # request も prompt_key_override もない場合は何もしない
        if request is None and prompt_key_override is None:
            return None

        # 使うプロンプトキーを決定
        prompt_key = prompt_key_override or (request.lower() if request is not None else None)
        if prompt_key is None or prompt_key not in self.config["prompt"]:
            return None

        # 必要に応じて少し待つ（API呼び出し間隔）
        if float(self.config["llm"].get("sleep_time", 0)) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))

        # プロンプトテンプレートをレンダリングして prompt_text を作る
        template_str = self.config["prompt"][prompt_key]
        key: dict[str, Any] = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
        }
        if extra_context:
            key.update(extra_context)

        template: Template = Template(template_str)
        prompt_text = template.render(**key).strip()

        # 使用する LLM と「全履歴ストア」を決定
        if llm_name in self.llm_models:
            llm_model = self.llm_models[llm_name]
            history_store = self.llm_message_histories.setdefault(llm_name, [])
        else:
            # フォールバック: どの llm_name も見つからない場合、最初の1つを使う
            if not self.llm_models:
                self.agent_logger.logger.error("LLM '%s' is not initialized", llm_name)
                return None
            fallback_name = next(iter(self.llm_models))
            llm_model = self.llm_models[fallback_name]
            history_store = self.llm_message_histories.setdefault(fallback_name, [])
            llm_name = fallback_name  # 実際に使った名前で上書き

        # 「LLM に見せる履歴（ベース）」を選ぶ
        if use_context_history:
            base_ctx = self.llm_context_histories.get(llm_name)
            if base_ctx is None:
                # まだコンテキストが無いときは、全履歴にフォールバック
                base_ctx = history_store
        else:
            base_ctx = history_store

        # LLM に渡す履歴（コピー）を組み立てる
        history_for_llm: list[BaseMessage] = list(base_ctx)
        history_for_llm.append(HumanMessage(content=prompt_text))

        try:
            # ここで実際に LLM を叩く
            response = (llm_model | StrOutputParser()).invoke(history_for_llm)

            # 全履歴には必ず今回のやりとりを積む
            history_store.append(HumanMessage(content=prompt_text))
            history_store.append(AIMessage(content=response))

            # 必要ならコンテキスト履歴にも積む
            if add_to_context_history:
                ctx = self.llm_context_histories.setdefault(llm_name, [])
                ctx.append(HumanMessage(content=prompt_text))
                ctx.append(AIMessage(content=response))

            # 既存のメインログ
            self.agent_logger.logger.info(["LLM", llm_name, prompt_text, response])

            # 追加: llm1 / llm2 ごとの専用ログ
            #   - llm2 → talk ログ
            #   - llm1 → commitment ログ
            try:
                if llm_name == "llm2":
                    self.agent_logger.log_llm_interaction(
                        kind="talk",
                        llm_name=llm_name,
                        prompt=prompt_text,
                        response=response,
                    )
                elif llm_name == "llm1":
                    self.agent_logger.log_llm_interaction(
                        kind="commitment",
                        llm_name=llm_name,
                        prompt=prompt_text,
                        response=response,
                    )
            except Exception:
                # ログ出力でコケてもゲーム自体は止めない
                self.agent_logger.logger.exception(
                    "Failed to write llm interaction log for %s",
                    llm_name,
                )

        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM '%s'", llm_name)
            return None
        else:
            return response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        llm_cfg = self.config["llm"]
        backends_cfg: dict[str, Any] | None = llm_cfg.get("backends")

        if backends_cfg:
            # ★ 新方式: llm.backends に複数定義されている場合
            for name, cfg in backends_cfg.items():
                model = self._create_llm_model(str(cfg["type"]), cfg)
                self.llm_models[name] = model
                self.llm_message_histories[name] = []
        else:
            # ★ 旧方式: 単一 LLM 設定
            model_type = str(llm_cfg["type"])
            base_cfg = self.config.get(model_type, {})
            cfg = {
                "model": base_cfg.get("model"),
                "temperature": base_cfg.get("temperature", 0.7),
                "base_url": base_cfg.get("base_url"),
            }
            model = self._create_llm_model(model_type, cfg)
            # 旧方式では "default" という名前で扱う
            self.llm_models["default"] = model
            self.llm_message_histories["default"] = []

        # ★ 各LLMバックエンドごとに initialize プロンプトを送信し、その履歴を保存
        for llm_name in self.llm_models.keys():
            # llmごとの initialize 用プロンプトキーを決める
            # - まず "initialize_<llm_name>" を探す（例: initialize_llm1, initialize_llm2）
            # - なければ共通の "initialize" を使う
            specific_key = f"initialize_{llm_name}"
            if specific_key in self.config["prompt"]:
                prompt_key = specific_key
            elif "initialize" in self.config["prompt"]:
                prompt_key = "initialize"
            else:
                # initialize系のプロンプトが何も定義されていない場合はスキップ
                continue

            # _send_message_to_llm を使って initialize を実行
            response = self._send_message_to_llm(
                request=None,
                llm_name=llm_name,
                prompt_key_override=prompt_key,
                extra_context=None,
                use_context_history=False,      # まだコンテキストは空なので False でOK
                add_to_context_history=True,    # initialize はコンテキストに含める
            )

            # initialize のやり取りを log/initialize/ 以下に保存
            # self._save_initialize_history(llm_name)

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(
            self.request,
            use_context_history=True,       # すでに作ってあるコンテキストを使う
            add_to_context_history=True,    # daily_initialize もコンテキストに含める
        )

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,       # initialize + daily_initialize だけを見る
            add_to_context_history=False,   # 囁き自体はコンテキストには足さない
        )
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    # def talk(self) -> str:
    #     """Return response to talk request.

    #     トークリクエストに対する応答を返す.

    #     Returns:
    #         str: Talk message / 発言メッセージ
    #     """
    #     response = self._send_message_to_llm(
    #         self.request,
    #         use_context_history=True,       # ★ initialize + daily_initialize だけをベースにする
    #         add_to_context_history=False,   # ★ talk 自体はコンテキストには足さない
    #     )
    #     self.sent_talk_count = len(self.talk_history)
    #     return response or ""

    # def talk(self) -> str:
    #     """Return response to talk request.

    #     トークリクエストに対する応答を返す.

    #     Returns:
    #         str: Talk message / 発言メッセージ
    #     """
    #     # ★ 1. 前回以降に追加された talk_history に対して A（talk_analysis）を実行
    #     try:
    #         self.last_analyzed_talk_index = run_talk_analysis_for_new_talks(
    #             agent=self,
    #             start_index=self.last_analyzed_talk_index,
    #         )
    #     except Exception:
    #         # 解析でこけてもゲーム自体は止めない（ログだけ残す）
    #         self.agent_logger.logger.exception(
    #             "Failed to run talk analysis (A) for agent %s",
    #             self.agent_name,
    #         )

    #     # ★ 2. いつも通り llm2 で発話を生成
    #     response = self._send_message_to_llm(
    #         self.request,
    #         use_context_history=True,       # initialize + daily_initialize だけをベースにする
    #         add_to_context_history=False,   # talk 自体はコンテキストには足さない
    #     )
    #     self.sent_talk_count = len(self.talk_history)
    #     return response or ""
    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        # ★ 1. 前回以降に追加された talk_history に対して A/B/C 分析を実行
        try:
            self.last_analyzed_talk_index = run_talk_analysis_for_new_talks(
                agent=self,
                start_index=self.last_analyzed_talk_index,
            )
        except Exception:
            # 解析でこけてもゲーム自体は止めない（ログだけ残す）
            self.agent_logger.logger.exception(
                "Failed to run talk analysis (A/B/C) for agent %s",
                self.agent_name,
            )

        # ★ 2. A/B/C の JSON から talk/agent CSV を再生成
        try:
            rebuild_csv_for_agent(self)
        except Exception:
            self.agent_logger.logger.exception(
                "Failed to rebuild CSV for agent %s",
                self.agent_name,
            )

        # ★ 3. いつも通り llm2 で発話を生成
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,       # initialize + daily_initialize だけをベースにする
            add_to_context_history=False,   # talk 自体はコンテキストには足さない
        )
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(
            self.request,
            use_context_history=True,       # すでに作ってあるコンテキストを使う
            add_to_context_history=True,    # daily_finish もコンテキストに含める
        )

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,        # initialize + daily_initialize + これまでの行動コンテキストを使う
            add_to_context_history=True,     # 今回の占い先決定もコンテキストに積む
        )
        return response or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,        # これまでのコンテキストを見ながら護衛先を決める
            add_to_context_history=True,     # 今回の護衛先決定もコンテキストに積む
        )
        return response or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,        # これまでの「方針」を見ながら投票先を決める
            add_to_context_history=True,     # 今回の投票先もコンテキストに積む
        )
        return response or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        response = self._send_message_to_llm(
            self.request,
            use_context_history=True,        # これまでのコンテキストを踏まえて襲撃先を決める
            add_to_context_history=True,     # 今回の襲撃先もコンテキストに積む
        )
        return response or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
