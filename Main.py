import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from llama_cpp import Llama
import time
import threading
import requests
import os

# Default configuration
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEFAULT_PORT = 8080

# App setup
app = FastAPI(title="TinyLlama Server")
model_lock = threading.Lock()
start_time = time.time()

# Load model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
)

# Request/Response schemas
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

# Prompt builder
def build_prompt(message: str, history: List[MessageEntry]) -> str:
    prompt = "<|system|>\nYou are Libra designed By Sanne Karibo, an AI expert market analyst and chatbot, answer questions from analysis.\n</s>\n"
    for msg in history:
        prompt += f"<|{msg.role}|>\n{msg.content}</s>\n"
    prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"
    return prompt

# Prediction function
def predict(prompt: str, max_tokens=512, temperature=0.7, top_p=0.5, repeat_penalty=1.1):
    with model_lock:
        start = time.time()
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["</s>", "<|user|>"],
        )
        response = output["choices"][0]["text"].strip()
        tokens_used = output["usage"]["completion_tokens"]
        elapsed = int((time.time() - start) * 1000)
        return response, tokens_used, elapsed

# Routes
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required")

    prompt = build_prompt(req.message, req.history or [])
    response, tokens, elapsed = predict(prompt)
    return ChatResponse(response=response, tokens=tokens, time_ms=elapsed)

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

# Start server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=DEFAULT_PORT, log_level="info")
