from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing OPENAI_API_KEY environment variable"
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="openai package not installed"
        )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
            ],
        )

        reply = response.choices[0].message.content

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {str(exc)}"
        )

    return {"reply": reply}



