import json
import os
import time
import uuid
import threading
import logging
import asyncio
import concurrent.futures
from typing import Any, List, Optional, Dict, Generator

from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from faker import Faker
import requests

# --- 基本配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 全局变量 ---
config = {}
account_manager: Optional["AccountManager"] = None
freeplay_client: Optional["FreeplayClient"] = None
valid_client_keys = set()
app_lock = threading.Lock()  # 用于保护全局资源的初始化


# --- Pydantic模型定义 (来自模板) ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.5  # 映射到Freeplay参数
    max_tokens: Optional[int] = 4096  # 映射到Freeplay参数
    top_p: Optional[float] = 1.0  # 映射到Freeplay参数


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "freeplay"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# --- 模型映射 ---
MODEL_MAPPING = {
    "claude-3-7-sonnet-20250219": {
        "model_id": "be71f37b-1487-49fa-a989-a9bb99c0b129",
        "max_tokens": 64000,
        "provider": "Anthropic",
    },
    "claude-4-opus-20250514": {
        "model_id": "bebc7dd5-a24d-4147-85b0-8f62902ea1a3",
        "max_tokens": 32000,
        "provider": "Anthropic",
    },
    "claude-4-sonnet": {
        "model_id": "884dde7c-8def-4365-b19a-57af2787ab84",
        "max_tokens": 64000,
        "provider": "Anthropic",
    },
}


# --- 服务类 ---
class FreeplayClient:
    def __init__(self, proxy_config: Optional[str] = None):
        self.proxies = {"http": proxy_config, "https": proxy_config} if proxy_config else None
        self.faker = Faker()

    def check_balance(self, session_id: str) -> float:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        cookies = {"session": session_id}
        try:
            response = requests.get(
                "https://app.freeplay.ai/app_data/settings/billing",
                headers=headers,
                cookies=cookies,
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                for feature in data.get("feature_usage", []):
                    if feature.get("feature_name") == "Freeplay credits":
                        return feature.get("usage_limit", 0) - feature.get(
                            "usage_value", 0
                        )
            return 0.0
        except Exception as e:
            logging.warning(
                f"Failed to check balance for session_id ending in ...{session_id[-4:]}: {e}"
            )
            return 0.0

    def register(self) -> Optional[Dict]:
        url = "https://app.freeplay.ai/app_data/auth/signup"
        payload = {
            "email": self.faker.email(),
            "password": f"aA1!{uuid.uuid4().hex[:8]}",
            "account_name": self.faker.name(),
            "first_name": self.faker.first_name(),
            "last_name": self.faker.last_name(),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "origin": "https://app.freeplay.ai",
            "referer": "https://app.freeplay.ai/signup",
        }
        try:
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                proxies=self.proxies,
                timeout=20,
            )
            response.raise_for_status()
            project_id = response.json()["project_id"]
            session = response.cookies.get("session")
            if project_id and session:
                return {
                    "email": payload["email"],
                    "password": payload["password"],
                    "session_id": session,
                    "project_id": project_id,
                    "balance": 5.0,  # 新注册账号默认5刀
                }
        except Exception as e:
            logging.error(f"Account registration failed: {e}")
            return None

    def chat(
        self,
        session_id: str,
        project_id: str,
        model_config: Dict,
        messages: List[Dict],
        params: Dict,
    ) -> requests.Response:
        url = f"https://app.freeplay.ai/app_data/projects/{project_id}/llm-completions"
        headers = {
            "accept": "*/*",
            "origin": "https://app.freeplay.ai",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        }
        cookies = {"session": session_id}

        # 兼容 system message
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)

        if system_prompt:
            # 将 system prompt 插入到第一个 user message 前
            if user_messages:
                user_messages[0][
                    "content"
                ] = f"{system_prompt}\n\nUser: {user_messages[0]['content']}"

        json_payload = {
            "messages": user_messages,
            "params": [
                {
                    "name": "max_tokens",
                    "value": params.get("max_tokens", model_config["max_tokens"]),
                    "type": "integer",
                },
                {
                    "name": "temperature",
                    "value": params.get("temperature", 0.5),
                    "type": "float",
                },
                {"name": "top_p", "value": params.get("top_p", 1.0), "type": "float"},
            ],
            "model_id": model_config["model_id"],
            "variables": {},
            "history": None,
            "asset_references": {},
        }
        files = {"json_data": (None, json.dumps(json_payload))}
        return requests.post(
            url, headers=headers, cookies=cookies, files=files, stream=True
        )


class AccountManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.accounts = []
        self.lock = threading.Lock()
        self.load_accounts()

    def load_accounts(self):
        with self.lock:
            if not os.path.exists(self.filepath):
                self.accounts = []
                return
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    self.accounts = json.load(f)
                except json.JSONDecodeError:
                    self.accounts = []
                    logging.warning(f"Could not decode JSON from {self.filepath}. Initializing with empty accounts.")
            logging.info(f"Loaded {len(self.accounts)} accounts from {self.filepath}")

    def save_accounts(self):
        with self.lock:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.accounts, f, indent=4)

    def add_account(self, account: Dict):
        with self.lock:
            self.accounts.append(account)
        self.save_accounts()
        logging.info(f"Added new account: {account.get('email')}")

    def get_account(self) -> Optional[Dict]:
        with self.lock:
            # 优先选择余额最高的
            available_accounts = [
                acc for acc in self.accounts if acc.get("balance", 0) > 0
            ]
            if not available_accounts:
                return None
            return max(available_accounts, key=lambda x: x.get("balance", 0))

    def update_account(self, account_data: Dict):
        with self.lock:
            for i, acc in enumerate(self.accounts):
                if acc["session_id"] == account_data["session_id"]:
                    self.accounts[i] = account_data
                    break
        self.save_accounts()

    def get_all_accounts(self) -> List[Dict]:
        with self.lock:
            return self.accounts.copy()


class KeyMaintainer(threading.Thread):
    def __init__(
        self, account_manager: AccountManager, client: FreeplayClient, config: Dict
    ):
        super().__init__(daemon=True)
        self.manager = account_manager
        self.client = client
        self.config = config

    def run(self):
        while True:
            try:
                logging.info("KeyMaintainer: Starting maintenance cycle.")
                accounts = self.manager.get_all_accounts()

                # 更新所有账户余额
                for account in accounts:
                    balance = self.client.check_balance(account["session_id"])
                    if balance != account.get("balance"):
                        account["balance"] = balance
                        self.manager.update_account(account)
                        logging.info(
                            f"Account {account['email']} balance updated to ${balance:.4f}"
                        )

                # 检查是否需要注册新账号
                healthy_accounts = [
                    acc
                    for acc in self.manager.get_all_accounts()
                    if acc.get("balance", 0) > self.config["LOW_BALANCE_THRESHOLD"]
                ]
                needed = self.config["ACTIVE_KEY_THRESHOLD"] - len(healthy_accounts)

                if needed > 0:
                    logging.info(
                        f"Healthy accounts ({len(healthy_accounts)}) below threshold ({self.config['ACTIVE_KEY_THRESHOLD']}). Need to register {needed} new accounts."
                    )
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.config["REGISTRATION_CONCURRENCY"]
                    ) as executor:
                        futures = [
                            executor.submit(self.client.register) for _ in range(needed)
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            new_account = future.result()
                            if new_account:
                                self.manager.add_account(new_account)
                else:
                    logging.info(
                        f"Sufficient healthy accounts ({len(healthy_accounts)}). No registration needed."
                    )

            except Exception as e:
                logging.error(f"Error in KeyMaintainer cycle: {e}")

            time.sleep(self.config["CHECK_INTERVAL_SECONDS"])


# --- FastAPI应用 ---
app = FastAPI(title="Freeplay.ai to OpenAI API Adapter")
security = HTTPBearer()


def initialize_app():
    global config, account_manager, freeplay_client, valid_client_keys

    with app_lock:
        if account_manager:  # 已经初始化
            return

        # 1. 加载配置
        if not os.path.exists("config.json"):
            default_config = {
                "HOST": "0.0.0.0",
                "PORT": 8000,
                "CLIENT_API_KEYS_FILE": "client_api_keys.json",
                "ACCOUNTS_FILE": "accounts.json",
                "LOW_BALANCE_THRESHOLD": 2.0,
                "ACTIVE_KEY_THRESHOLD": 5,
                "CHECK_INTERVAL_SECONDS": 300,
                "REGISTRATION_CONCURRENCY": 2,
                "REGISTRATION_PROXY": None,  # 例如 "http://user:pass@host:port"
            }
            with open("config.json", "w") as f:
                json.dump(default_config, f, indent=4)
            config = default_config
            logging.info("Created default config.json")
        else:
            with open("config.json", "r") as f:
                config = json.load(f)
            logging.info("Loaded config from config.json")

        # 2. 加载客户端密钥
        if not os.path.exists(config["CLIENT_API_KEYS_FILE"]):
            dummy_key = f"sk-freeplay-{uuid.uuid4().hex}"
            with open(config["CLIENT_API_KEYS_FILE"], "w") as f:
                json.dump([dummy_key], f, indent=4)
            valid_client_keys = {dummy_key}
            logging.info(f"Created dummy client_api_keys.json with key: {dummy_key}")
        else:
            with open(config["CLIENT_API_KEYS_FILE"], "r") as f:
                valid_client_keys = set(json.load(f))
            logging.info(f"Loaded {len(valid_client_keys)} client API keys.")

        # 3. 初始化服务
        freeplay_client = FreeplayClient(proxy_config=config.get("REGISTRATION_PROXY"))
        account_manager = AccountManager(filepath=config["ACCOUNTS_FILE"])

        # 4. 启动后台维护线程
        maintainer = KeyMaintainer(account_manager, freeplay_client, config)
        maintainer.start()
        logging.info("Key maintenance service started.")


async def authenticate_client(auth: HTTPAuthorizationCredentials = Depends(security)):
    if not auth or auth.credentials not in valid_client_keys:
        raise HTTPException(status_code=403, detail="Invalid client API key.")


@app.on_event("startup")
async def startup_event():
    initialize_app()


@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(authenticate_client)):
    model_infos = [
        ModelInfo(id=name, owned_by=details["provider"])
        for name, details in MODEL_MAPPING.items()
    ]
    return ModelList(data=model_infos)


def stream_generator(
    response: requests.Response, model_name: str, account: Dict
) -> Generator[str, None, None]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # Start chunk
    start_chunk = StreamResponse(
        model=model_name, choices=[StreamChoice(delta={"role": "assistant"})]
    ).dict()
    start_chunk["id"] = chat_id
    start_chunk["created"] = created
    yield f"data: {json.dumps(start_chunk)}\n\n"

    try:
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("content"):
                        chunk = StreamResponse(
                            model=model_name,
                            choices=[StreamChoice(delta={"content": data["content"]})],
                        ).dict()
                        chunk["id"] = chat_id
                        chunk["created"] = created
                        yield f"data: {json.dumps(chunk)}\n\n"
                    if data.get("cost") is not None:
                        break  # 结束
                except json.JSONDecodeError:
                    continue
    finally:
        # End chunk
        end_chunk = StreamResponse(
            model=model_name, choices=[StreamChoice(delta={}, finish_reason="stop")]
        ).dict()
        end_chunk["id"] = chat_id
        end_chunk["created"] = created
        yield f"data: {json.dumps(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        # 更新余额
        assert freeplay_client is not None
        assert account_manager is not None
        new_balance = freeplay_client.check_balance(account["session_id"])
        if new_balance != account.get("balance"):
            account["balance"] = new_balance
            account_manager.update_account(account)
            logging.info(
                f"Post-chat balance update for {account['email']}: ${new_balance:.4f}"
            )


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    if req.model not in MODEL_MAPPING:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found.")

    model_config = MODEL_MAPPING[req.model]
    messages_dict = [msg.dict() for msg in req.messages]

    # 账户选择和重试逻辑
    assert account_manager is not None
    max_retries = len(account_manager.get_all_accounts())
    for attempt in range(max_retries):
        account = account_manager.get_account()
        if not account:
            raise HTTPException(
                status_code=503, detail="No available accounts in the pool."
            )

        try:
            params = {
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
            }
            assert freeplay_client is not None
            response = freeplay_client.chat(
                account["session_id"],
                account["project_id"],
                model_config,
                messages_dict,
                params,
            )

            if response.status_code == 200:
                # 请求成功
                if req.stream:
                    return StreamingResponse(
                        stream_generator(response, req.model, account),
                        media_type="text/event-stream",
                    )
                else:
                    full_content = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                full_content += data.get("content", "")
                                if data.get("cost") is not None:
                                    break
                            except json.JSONDecodeError:
                                continue

                    # 更新余额
                    assert freeplay_client is not None
                    assert account_manager is not None
                    new_balance = freeplay_client.check_balance(account["session_id"])
                    account["balance"] = new_balance
                    account_manager.update_account(account)
                    logging.info(
                        f"Post-chat balance update for {account['email']}: ${new_balance:.4f}"
                    )

                    return ChatCompletionResponse(
                        model=req.model,
                        choices=[
                            ChatCompletionChoice(
                                message=ChatMessage(
                                    role="assistant", content=full_content
                                )
                            )
                        ],
                    )

            elif response.status_code in [401, 403, 404]:
                logging.warning(
                    f"Account {account['email']} failed with status {response.status_code}. Disabling it."
                )
                account["balance"] = 0.0  # 禁用账户
                assert account_manager is not None
                account_manager.update_account(account)
                continue  # 重试下一个
            else:
                logging.error(
                    f"API call failed with status {response.status_code}: {response.text}"
                )
                response.raise_for_status()

        except Exception as e:
            logging.error(
                f"Error with account {account['email']}: {e}. Trying next account."
            )
            account["balance"] = 0.0  # 发生未知异常也禁用
            assert account_manager is not None
            account_manager.update_account(account)
            continue

    raise HTTPException(
        status_code=503, detail="All available accounts failed to process the request."
    )


@app.get("/admin/accounts/status")
async def accounts_status(_: None = Depends(authenticate_client)):
    assert account_manager is not None
    accounts = account_manager.get_all_accounts()
    total_balance = sum(acc.get("balance", 0) for acc in accounts)
    healthy_count = len(
        [
            acc
            for acc in accounts
            if acc.get("balance", 0) > config.get("LOW_BALANCE_THRESHOLD", 2.0)
        ]
    )

    return JSONResponse(
        {
            "total_accounts": len(accounts),
            "healthy_accounts": healthy_count,
            "total_balance": f"${total_balance:.4f}",
            "accounts": [
                {
                    "email": acc.get("email"),
                    "balance": f"${acc.get('balance', 0):.4f}",
                    "project_id": acc.get("project_id"),
                }
                for acc in accounts
            ],
        }
    )


if __name__ == "__main__":
    import uvicorn

    initialize_app()
    logging.info("--- Freeplay.ai to OpenAI API Adapter ---")
    logging.info(f"Starting server on {config['HOST']}:{config['PORT']}")
    logging.info(f"Supported models: {list(MODEL_MAPPING.keys())}")
    logging.info(f"Client keys loaded: {len(valid_client_keys)}")
    assert account_manager is not None
    logging.info(f"Accounts loaded: {len(account_manager.get_all_accounts())}")
    logging.info("Endpoints:")
    logging.info("  POST /v1/chat/completions (Client API Key Auth)")
    logging.info("  GET  /v1/models (Client API Key Auth)")
    logging.info("  GET  /admin/accounts/status (Client API Key Auth)")

    uvicorn.run(app, host=config["HOST"], port=config["PORT"])