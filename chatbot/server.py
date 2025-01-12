from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

# Assuming the ConversationManager class is defined in conversation_manager.py
from ConversationManager import ConversationManager

app = FastAPI()

class UserMessage(BaseModel):
    thread_id: str
    user_message: str

# Load model parameters and initialize ConversationManager
model_params = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.2,
    "max_tokens": None,
    "timeout": None,
    "max_retries": 2,
    "safety_settings": {
        "HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT": "HarmBlockThreshold.BLOCK_NONE",
    },
}
file_name = './ali_cv_markdown.md'

conversation_manager = ConversationManager(model_params, file_name)

@app.post("/chat/handle")
def handle_conversation(user_message: UserMessage):
    try:
        # Invoke the conversation handler
        response = conversation_manager.handle_conversation(user_message.thread_id, user_message.user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5455)
