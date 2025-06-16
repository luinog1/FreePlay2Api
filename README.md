# FreePlay2Api

这是一个将 Freeplay.ai 接口转换为 OpenAI API 兼容接口的适配器，方便您使用习惯的 OpenAI 客户端与 Freeplay.ai 进行交互。

## 安装与运行

请按照以下步骤设置和运行项目：

1.  **安装 `uv` (推荐)**

    如果您尚未安装 `uv`，可以通过 PowerShell 运行以下命令进行安装：

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **创建并激活虚拟环境**

    使用 `uv` 创建并激活 Python 3.13 的虚拟环境：

    ```bash
    uv python install 3.13
    uv venv
    .venv\Scripts\activate
    ```

3.  **安装依赖**

    激活虚拟环境后，安装项目所需依赖：

    ```bash
    uv sync
    ```

4.  **运行应用**

    ```bash
    python main.py
    ```

    应用将默认运行在 `http://0.0.0.0:8000`。

## 配置

首次运行会生成 `config.json` 和 `client_api_keys.json` 文件，您可以根据需要使用它们。

*   `config.json`: 包含端口、账户文件路径、代理设置等。
*   `client_api_keys.json`: 包含用于访问适配器的客户端 API 密钥。

## API 端点

*   `POST /v1/chat/completions` (需要客户端 API 密钥认证)
*   `GET /v1/models` (需要客户端 API 密钥认证)
*   `GET /admin/accounts/status` (需要客户端 API 密钥认证)
