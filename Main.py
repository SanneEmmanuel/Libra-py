import os
import time
import threading
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from llama_cpp import Llama

# -------------------- Config --------------------
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_DIR = "/tmp"
MODEL_PATH = os.path.join(MODEL_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

CONTEXT_SIZE = 2048
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.5
REPEAT_PENALTY = 1.1

# -------------------- Ensure Model --------------------
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("[🔁] Downloading TinyLlama model...")
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("[✅] Model downloaded to:", MODEL_PATH)

# -------------------- App + Schema --------------------
app = FastAPI(title="Libra AI - TinyLlama")
model_lock = threading.Lock()
start_time = time.time()

class MessageEntry(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[MessageEntry]] = []

class ChatResponse(BaseModel):
    response: str
    tokens: int
    time_ms: int

class PingRequest(BaseModel):
    callback_url: HttpUrl

# -------------------- Load Model --------------------
ensure_model_exists()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_threads=4,
    n_gpu_layers=0,
)

# -------------------- Core Logic --------------------
def build_prompt(message: str, history: List[MessageEntry]) -> str:
    prompt = "<|system|>\nYou are Libra designed By Sanne Karibo, an AI expert market analyst and chatbot, answer questions from analysis.\n</s>\n"
    for msg in history:
        prompt += f"<|{msg.role}|>\n{msg.content}</s>\n"
    prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"
    return prompt

def predict(prompt: str) -> tuple[str, int, int]:
    with model_lock:
        start = time.time()
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            stop=["</s>", "<|user|>"],
        )
        response = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        duration_ms = int((time.time() - start) * 1000)
        return response, tokens, duration_ms

# -------------------- Routes --------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required")
    prompt = build_prompt(req.message, req.history or [])
    response, tokens, time_ms = predict(prompt)
    return ChatResponse(response=response, tokens=tokens, time_ms=time_ms)

@app.post("/ping")
async def ping(req: PingRequest, bg: BackgroundTasks):
    def delayed_callback(url: str):
        time.sleep(8 * 60)
        try:
            requests.post(url, json={"status": "pong"})
        except Exception:
            pass
    bg.add_task(delayed_callback, req.callback_url)
    return {"message": "pong scheduled"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime": f"{int(time.time() - start_time)}s",
        "timestamp": int(time.time())
    }

# -------------------- Uvicorn Entrypoint --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info")
