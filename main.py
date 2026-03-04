from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import httpx
from azure.storage.blob import BlobServiceClient, ContentSettings

from database import get_db, engine
import models

app = FastAPI()

# --- [0. AI 모델 설정값] ---
CUSTOM_VISION_URL = "https://gdscs-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/720991f1-25e4-4d32-968d-0e00abbb1166/classify/iterations/Iteration5/image"
CUSTOM_VISION_KEY = "***REMOVED***"

# --- [1. Azure Blob Storage 설정] ---
AZURE_STORAGE_CONNECTION_STRING = "***REMOVED***"
CONTAINER_NAME = "images" 

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Storage 연결 설정 실패: {e}")

# --- [AI 분석 보조 함수들] ---

async def analyze_text_ai(text: str):
    if not text: return {"label": "safe", "score": 0.0}
    return {"label": "safe", "score": 0.05}

async def analyze_image_ai(image_bytes: bytes):
    """Custom Vision 이미지 모델 실제 호출"""
    if not image_bytes: 
        return {"label": "no_image", "probability": 0.0}
    
    try:
        headers = {
            "Prediction-Key": CUSTOM_VISION_KEY,
            "Content-Type": "application/octet-stream"
        }
        
        # 💡 httpx 사용 시 좀 더 명확하게 content로 전달
        async with httpx.AsyncClient() as client:
            response = await client.post(
                CUSTOM_VISION_URL, 
                content=image_bytes, # 바이너리 데이터 직접 전송
                headers=headers, 
                timeout=10.0
            )
            
        if response.status_code == 200:
            result = response.json()
            if result.get('predictions'):
                top_prediction = result['predictions'][0]
                return {
                    "label": top_prediction['tagName'], 
                    "probability": top_prediction['probability']
                }
            return {"label": "unknown", "probability": 0.0}
        else:
            # 💡 에러 발생 시 상세 내용을 서버 터미널에 출력
            print(f"⚠️ [IMAGE AI] API 에러 ({response.status_code}): {response.text}")
            return {"label": "error", "probability": 0.0}
            
    except Exception as e:
        print(f"❌ [IMAGE AI] 예외 발생: {str(e)}")
        return {"label": "error", "probability": 0.0}

async def upload_image_to_blob(contents: bytes, filename: str, content_type: str):
    """Refactored: 이미 읽은 바이트 데이터를 받아 업로드하도록 변경"""
    try:
        ext = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=unique_filename)
        blob_client.upload_blob(
            contents, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 실패: {e}")
        return None

# --- [3. 댓글 로직 수정본] ---
@app.post("/comments")
async def create_comment(
    content: Optional[str] = Form(None),
    post_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        if not content and not image:
            raise HTTPException(status_code=400, detail="내용이나 이미지 중 하나는 필수입니다.")

        text_ai_res = {"label": "safe", "score": 0.0}
        image_ai_res = {"label": "clean", "probability": 0.0}
        uploaded_url = None

        # 1. 텍스트 분석
        if content:
            text_ai_res = await analyze_text_ai(content)

        # 2. 이미지 처리 (분석 + 업로드)
        if image:
            # 💡 파일을 한 번만 읽어서 두 곳(AI 분석, Blob 업로드)에 사용합니다.
            image_data = await image.read()
            
            # AI 분석 호출
            image_ai_res = await analyze_image_ai(image_data)
            
            # 업로드 호출 (이미 읽은 image_data 사용)
            uploaded_url = await upload_image_to_blob(image_data, image.filename, image.content_type)

        # 3. 판별 라벨 결정
        final_label = text_ai_res["label"] 
        if image_ai_res["label"].lower() != "clean" and image_ai_res["label"].lower() != "no_image":
            if image_ai_res["probability"] > 0.6:
                final_label = f"unsafe_image_{image_ai_res['label']}"

        # 4. DB 저장
        new_comment = models.Comment(
            post_id=post_id,
            user_id=6,
            content=content if content else "",
            image_url=uploaded_url,
            toxicity_score=text_ai_res["score"],
            label=final_label
        )
        
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        
        # 💡 성공 응답 반환 (JSON 구조 유지)
        return {
            "status": "success", 
            "image_url": uploaded_url, 
            "ai_result": {
                "text": text_ai_res, 
                "image": image_ai_res
            }
        }
        
    except Exception as e:
        db.rollback()
        print(f"🔥 서버 에러 상세: {str(e)}") # 👈 에러 원인을 터미널에 출력
        # 💡 브라우저에 500 에러와 함께 원인 메시지를 보냅니다.
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "detail": str(e)}
        )

# --- 나머지 기존 코드 (Post 로직, Jinja2 등) 유지 ---
@app.get("/posts")
async def get_posts(db: Session = Depends(get_db)):
    posts = db.query(models.Post).order_by(models.Post.id.desc()).all()
    result = []
    for post in posts:
        comment_list = []
        for c in post.comments:
            comment_list.append({
                "id": c.id, "content": c.content, "image_url": c.image_url,
                "username": c.author.username if c.author else "익명",
                "role": c.author.role if c.author else "user"
            })
        result.append({
            "id": post.id, "body": post.body,
            "username": post.author.username if post.author else "익명",
            "role": post.author.role if post.author else "user",
            "created_at": post.created_at.isoformat() if post.created_at else None,
            "comments": comment_list
        })
    return result

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})