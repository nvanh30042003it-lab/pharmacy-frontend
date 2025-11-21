from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

router = APIRouter(prefix="/ai", tags=["AI Chat"])


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = """
Bạn là chatbot tư vấn thuốc của Nhà Thuốc ANHDUONG.
Bạn chỉ trả lời ngắn gọn, lịch sự, dễ hiểu, không tư vấn y tế chuyên sâu.
"""


@router.post("/chat")
def chat_ai(req: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
            ],
        )

        reply = response.output_text

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"reply": reply}
