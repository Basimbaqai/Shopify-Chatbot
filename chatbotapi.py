from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import time
from langchain_chatbot import chat_with_streaming_async
from langchain_chatbot import initialize_vector_store
import json
import uvicorn

class ChatRequest(BaseModel):
    message: str


import traceback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events with robust error handling."""
    try:
        if initialize_vector_store():
            print("✅ Vector store initialized successfully!")
        else:
            raise RuntimeError("Failed to initialize vector store. The application cannot start.")

    except Exception as e:
        # This will catch any exception during initialization and print the full details.
        print("❌❌❌ AN ERROR OCCURRED DURING STARTUP ❌❌❌")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-------------------")
        raise e

    print("🎉 API is ready to serve requests!")
    yield
    print("👋 API shutting down...")
last_request_time = datetime.now()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "✅ Shopify Chatbot API is running!"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    global last_request_time

    user_input = req.message.strip()
    if not user_input:
        return {"error": "Empty message."}

    if user_input.lower() in ['quit', 'exit', 'bye']:
        return {"response": "👋 Goodbye!"}

    time_since_last = (datetime.now() - last_request_time).total_seconds()
    if time_since_last < 1:
        time.sleep(1 - time_since_last)
    last_request_time = datetime.now()

    try:
        async def stream_response():
            async for chunk in chat_with_streaming_async(user_input):
                data = {"chunk": chunk, "done": False}
                yield f"data: {json.dumps(data)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)