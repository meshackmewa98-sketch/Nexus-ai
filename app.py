import os
import json
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError

app = FastAPI(title="ChatGPT-Style Nexus AI Backend")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str  # 'user' or 'model'
    parts: list[dict]

class StreamChatRequest(BaseModel):
    contents: list[ChatMessage]
    model: str = "gemini-2.5-flash"

@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: StreamChatRequest,
    x_api_key: str | None = Header(default=None)
):
    api_key = x_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing Gemini API Key. Pass X-API-Key header or set GEMINI_API_KEY environment variable."
        )

    try:
        client = genai.Client(api_key=api_key)

        def event_generator():
            # Request token streaming from Gemini
            response_stream = client.models.generate_content_stream(
                model=request.model,
                contents=[c.model_dump() for c in request.contents]
            )
            for chunk in response_stream:
                if chunk.text:
                    # Format as SSE (Server-Sent Event)
                    data = json.dumps({"text": chunk.text})
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Points to app:app since the file name is app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
