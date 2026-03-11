# Vercel AI Gateway — Python SDK 版本業務流程與架構分析

本文件將原始 TypeScript / Deno 版本的 `src/main.ts` 閘道器邏輯，重新以 **Python (FastAPI)** 的視角進行架構分析與對照。
目標是為日後需要以 Python 實作 **OpenAI 相容代理閘道器 (Gateway)** 的開發者提供完整的參考藍圖。

> **核心差異摘要**：TypeScript 版本使用 Vercel AI SDK (`generateText` / `streamText`) + `@ai-sdk/gateway`；
> Python 版本需組合 **FastAPI** + **httpx** (非同步 HTTP) + **fastapi-ai-sdk** (SSE 串流格式化) 來實現相同的功能。

---

## 1. 核心職責與架構概覽

閘道器的核心目標不變：「**對外提供偽裝成 OpenAI API 格式的端點，對內呼叫 Vercel AI Gateway / 實際大語言模型 API**」。

### 核心依賴 (Python)

| TypeScript (原始) | Python (對應) | 用途 |
|:---|:---|:---|
| `generateText` from `'ai'` | `httpx.AsyncClient.post()` | 非串流文字生成（一次性取得完整回應） |
| `streamText` from `'ai'` | `httpx.AsyncClient.stream("POST", ...)` | 串流文字生成（逐 chunk 讀取） |
| `createGateway` from `'@ai-sdk/gateway'` | 手動建構 URL + Headers | 建立對 Vercel AI Gateway 上游的請求底層設定 |
| Deno `ReadableStream` + SSE 手動封裝 | `fastapi.responses.StreamingResponse` + `fastapi-ai-sdk` (`AIStream`, SSE Events) | SSE 串流回傳 |
| `Deno.serve()` | `uvicorn` + `FastAPI()` | ASGI 伺服器進入點 |

### 基本流程 (Python 版)

```python
# 1. 接收請求：FastAPI 路由
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    ...

# 2. 認證過濾：Depends 注入或中介層
api_key = get_bearer_token(request.headers.get("authorization"))

# 3. 路由分發：FastAPI 原生路由裝飾器
@app.get("/health")
@app.get("/v1/models")
@app.post("/v1/chat/completions")
@app.post("/v1/images/generations")

# 4. 格式轉換與執行：httpx 呼叫上游 + 回應包裝
```

---

## 2. 路由與端點解析

### 2.1 健康檢查與模型列表

```python
from fastapi import FastAPI
import os, time

app = FastAPI()
DEFAULT_MODEL = "google/gemini-3-pro-image"

@app.get("/")
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "openai-compat-vercel-gateway-python",
        "model": os.getenv("DEFAULT_MODEL", DEFAULT_MODEL),
        "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

@app.get("/v1/models")
async def list_models():
    raw = os.getenv("OPENAI_COMPAT_MODELS", "")
    model_ids = [m.strip() for m in raw.split(",") if m.strip()] \
                or [os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)]
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "vercel-ai-gateway"}
            for mid in model_ids
        ],
    }
```

### 2.2 對話補全 (`POST /v1/chat/completions`)

Python 版處理函數同樣包含**三條分支**：

#### 分支 1：包含工具呼叫的請求 — 透傳模式

```python
import httpx

async def proxy_chat_completion(body: dict, model_id: str, api_key: str):
    """繞過 SDK，直接轉發至上游 (Vercel AI Gateway)"""
    upstream_url = resolve_upstream_url()
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    if body.get("stream"):
        #串流透傳
        return StreamingResponse(
            stream_proxy(upstream_url, headers, body),
            media_type="text/event-stream",
        )
    else:
        #非串流透傳
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(upstream_url, json=body, headers=headers)
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")
```

> **參考價值**：Python 中使用 `httpx.AsyncClient.stream()` 實作透傳代理，比 `requests` 更適合非同步場景。

#### 分支 2：流式輸出請求 (`stream: true`)

```python
import json, uuid, asyncio
from fastapi.responses import StreamingResponse

async def handle_chat_stream(body: dict, model_id: str, api_key: str):
    """使用 httpx 串流讀取上游，並包裝成 OpenAI SSE 格式"""

    async def sse_generator():
        response_id = f"chatcmpl_{uuid.uuid4()}"
        created = int(time.time())

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                resolve_upstream_url(),
                json={**body, "model": model_id, "stream": True},
                headers={"authorization": f"Bearer {api_key}",
                         "content-type": "application/json"},
            ) as resp:
                async for chunk in resp.aiter_text():
                    #將上游 chunk 轉換為 OpenAI SSE 格式
                    sse_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(sse_chunk)}\n\n"

        #結束標記
        yield f"data: {json.dumps({**sse_chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache, no-transform"})
```

> **設計要點**：Python 版的 SSE 格式化核心與 TypeScript 一致 —— 必須遵守 `data: {...}\n\n` 標準格式，最後送出 `data: [DONE]\n\n` 結束串流。

#### 分支 3：一般文字請求 (`stream: false`)

```python
async def handle_chat_generate(body: dict, model_id: str, api_key: str):
    """非串流：一次性取得完整回應"""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            resolve_upstream_url(),
            json={**body, "model": model_id, "stream": False},
            headers={"authorization": f"Bearer {api_key}",
                     "content-type": "application/json"},
        )
        result = resp.json()

    #包裝成 OpenAI chat.completion 回應
    return {
        "id": f"chatcmpl_{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.get("text", "")},
            "finish_reason": "stop",
        }],
        "usage": to_openai_usage(result.get("usage")),
    }
```

### 2.3 圖片生成 (`POST /v1/images/generations`)

```python
async def handle_image_generations(body: dict, model_id: str, api_key: str):
    """文生圖端點：迴圈呼叫取得多張圖片"""
    count = min(max(body.get("n", 1), 1), 4)
    data = []

    for i in range(count):
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                resolve_upstream_url(),
                json={"model": model_id, "prompt": body["prompt"],
                      "providerOptions": build_provider_options(body)},
                headers={"authorization": f"Bearer {api_key}",
                         "content-type": "application/json"},
            )
            result = resp.json()
            images = extract_images(result.get("files", []))
            if not images:
                raise HTTPException(502, "模型未回傳圖片")

            img = images[0]
            if body.get("response_format") == "url":
                data.append({"url": f"data:{img['media_type']};base64,{img['base64']}"})
            else:
                data.append({"b64_json": img["base64"]})

    return {"created": int(time.time()), "data": data}
```

---

## 3. 關鍵的資料結構轉換 (Adapter Pattern)

### 3.1 訊息轉換 (`to_ai_sdk_messages` / `to_ai_sdk_part`)

```python
def to_ai_sdk_messages(messages: list[dict]) -> list[dict]:
    """將 OpenAI 多模態訊息轉為 Vercel AI SDK 格式"""
    result = []
    for msg in messages:
        role = normalize_role(msg.get("role", "user"))
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = [to_ai_sdk_part(p) for p in content]
            result.append({"role": role, "content": [p for p in parts if p]})
        else:
            result.append({"role": role, "content": ""})
    return result

def to_ai_sdk_part(part: dict) -> dict | None:
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    if part.get("type") == "image_url":
        url = (part.get("image_url") or {}).get("url")
        if url:
            return {"type": "image", "image": url}
    if part.get("type") == "file":
        data = part.get("file_data") or part.get("data") or part.get("url")
        if data:
            return {"type": "file", "data": data,
                    **({"mediaType": part["media_type"]} if "media_type" in part else {})}
    return None
```

### 3.2 供應商客製化參數 (`build_provider_options`)

```python
def build_provider_options(body: dict, image_settings: dict = None,
                           response_modalities: list = None) -> dict | None:
    """組裝 Vercel providerOptions — 巢狀合併 Google 圖片設定"""
    provider_opts = dict(body.get("providerOptions") or {})
    google_opts = dict(provider_opts.get("google") or {})
    image_config = dict(google_opts.get("imageConfig") or {})

    if image_settings:
        if image_settings.get("aspect_ratio"):
            image_config["aspectRatio"] = image_settings["aspect_ratio"]
        if image_settings.get("image_size"):
            image_config["imageSize"] = image_settings["image_size"]

    if image_config:
        google_opts["imageConfig"] = image_config
    if response_modalities:
        google_opts["responseModalities"] = response_modalities
    if google_opts:
        provider_opts["google"] = google_opts

    return provider_opts or None
```

> **設計巧思**：`find_scoped_image_config()` 會從 `providerOptions` 的多層巢狀結構中遞迴查找 `imageConfig`，提高與不同前端傳參結構的相容度。

---

## 4. 驗證與金鑰管理機制

```python
import os
from fastapi import Request, HTTPException

def check_inbound_api_key(request: Request) -> str | None:
    """驗證入站 API Key 並解析上游金鑰"""
    token = get_bearer_token(request.headers.get("authorization"))
    expected = os.getenv("OPENAI_COMPAT_API_KEY")

    if expected and token != expected:
        raise HTTPException(401, detail="Unauthorized")

    #金鑰透傳至下游供應商
    return resolve_upstream_api_key(token)

def resolve_upstream_api_key(inbound_key: str | None) -> str:
    compat_key = os.getenv("OPENAI_COMPAT_API_KEY")
    if compat_key:
        configured = os.getenv("AI_GATEWAY_API_KEY")
        if not configured:
            raise HTTPException(500, "啟用 OPENAI_COMPAT_API_KEY 時必須設定 AI_GATEWAY_API_KEY")
        return configured
    return inbound_key or os.getenv("AI_GATEWAY_API_KEY") or ""
```

---

## 5. SSE Keep-Alive 與逾時機制 (Python 實作)

TypeScript 版本使用 `setTimeout` 搭配 `ReadableStream` 實現 SSE keep-alive；
Python 版需要使用 `asyncio.create_task` + `asyncio.Event` 來實現相同的效果：

```python
import asyncio

async def sse_with_keepalive(upstream_task, keepalive_interval: float = 15.0):
    """帶 keep-alive 的 SSE async generator"""
    done = asyncio.Event()
    result_holder = {"data": None, "error": None}

    async def run_upstream():
        try:
            result_holder["data"] = await upstream_task()
        except Exception as e:
            result_holder["error"] = e
        finally:
            done.set()

    task = asyncio.create_task(run_upstream())

    #持續發送 keep-alive 直到上游完成
    while not done.is_set():
        try:
            await asyncio.wait_for(done.wait(), timeout=keepalive_interval)
        except asyncio.TimeoutError:
            yield ": keep-alive\n\n"  #SSE 註解格式

    #上游完成，輸出實際資料
    if result_holder["error"]:
        yield f"data: {json.dumps({'error': str(result_holder['error'])})}\n\n"
    else:
        #分塊發送大內容（如 Base64 圖片）
        content = result_holder["data"]
        chunk_size = 16384
        for i in range(0, len(content), chunk_size):
            yield f"data: {json.dumps(create_chunk(content[i:i+chunk_size]))}\n\n"

    yield "data: [DONE]\n\n"
```

---

## 6. 完整的 FastAPI 應用骨架

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx, os, uuid, time, json

app = FastAPI(title="Vercel AI Gateway (Python)")

DEFAULT_MODEL = "google/gemini-3-pro-image"
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT_MS", "120000")) / 1000

@app.get("/health")
async def health(): ...

@app.get("/v1/models")
async def list_models(): ...

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    api_key = check_inbound_api_key(request)
    model_id = resolve_model_id(body.get("model"))

    if has_tool_calling(body):
        return await proxy_chat_completion(body, model_id, api_key)
    if body.get("stream"):
        return await handle_chat_stream(body, model_id, api_key)
    return await handle_chat_generate(body, model_id, api_key)

@app.post("/v1/images/generations")
async def image_generations(request: Request):
    body = await request.json()
    api_key = check_inbound_api_key(request)
    return await handle_image_generations(body, resolve_model_id(body.get("model")), api_key)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
```

---

## 7. TypeScript → Python 對照速查表

| 概念 | TypeScript (Deno) | Python (FastAPI) |
|:---|:---|:---|
| 進入點 | `Deno.serve()` | `uvicorn.run(app)` |
| 路由 | `if (url.pathname === ...)` | `@app.get()` / `@app.post()` |
| 環境變數 | `Deno.env.get("KEY")` | `os.getenv("KEY")` |
| 非同步 HTTP | `fetch()` | `httpx.AsyncClient` |
| JSON 解析 | `request.json()` | `await request.json()` |
| SSE 串流 | `new ReadableStream()` + 手動 `enqueue` | `StreamingResponse(async_generator())` |
| UUID | `crypto.randomUUID()` | `uuid.uuid4()` |
| 逾時控制 | `AbortController` + `setTimeout` | `httpx.Timeout` + `asyncio.wait_for` |
| Keep-Alive | `setInterval` → `controller.enqueue(": keep-alive")` | `asyncio.create_task` → `yield ": keep-alive\n\n"` |
| 錯誤處理 | `try/catch` + `HttpError` class | `try/except` + `HTTPException` |
| Base64 轉換 | `btoa()` / `Buffer.from()` | `base64.b64encode()` |
| 型別系統 | TypeScript interfaces | Pydantic models / TypedDict |
