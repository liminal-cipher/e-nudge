from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form, status
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
    # 실제 텍스트 분석 로직이 필요하다면 여기에 추가 (현재는 기본값 반환)
    if not text: return {"label": "safe", "score": 0.0}
    return {"label": "safe", "score": 0.05}

async def analyze_image_ai(image_bytes: bytes):
    if not image_bytes: 
        return {"label": "no_image", "probability": 0.0}
    
    try:
        headers = {
            "Prediction-Key": CUSTOM_VISION_KEY,
            "Content-Type": "application/octet-stream"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                CUSTOM_VISION_URL, 
                content=image_bytes, 
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
            print(f"⚠️ [IMAGE AI] API 에러 ({response.status_code}): {response.text}")
            return {"label": "error", "probability": 0.0}
            
    except Exception as e:
        print(f"❌ [IMAGE AI] 예외 발생: {str(e)}")
        return {"label": "error", "probability": 0.0}

async def upload_image_to_blob(contents: bytes, filename: str, content_type: str):
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

# --- [2. 게시글(Post) 로직 추가] ---
class PostCreate(BaseModel):
    body: str

# --- [2. 게시글(Post) 로직 수정] ---

@app.post("/posts")
async def create_post(
    title: str = Form("제목 없음"),      # 추가: models.py에 title이 NVARCHAR로 있으므로 받아주는게 좋습니다.
    content: str = Form(...),          # 프론트에서 보내는 필드명
    db: Session = Depends(get_db)
):
    try:
        # models.py의 Post 클래스 구조에 맞게 매핑
        new_post = models.Post(
            title=title,
            body=content,              # 프론트의 content를 DB의 body 필드에 저장
            user_id=6,                 # 테스트용 유저 ID
            status="active"
        )
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        return new_post
    except Exception as e:
        db.rollback()
        print(f"🔥 게시글 저장 에러: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

# --- [3. 댓글 로직 수정본] ---
@app.post("/comments")
async def create_comment(
    content: Optional[str] = Form(None),
    post_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        # 1. 기본 분석 결과 초기화
        text_ai_res = {"label": "safe", "score": 0.0}
        image_ai_res = {"label": "clean", "probability": 0.0} # 기본값은 깨끗함
        uploaded_url = None

        # 2. 텍스트가 있을 때만 분석
        if content:
            text_ai_res = await analyze_text_ai(content)

        # 3. 이미지 처리 (이미지가 실제로 들어왔을 때만!)
        # image.filename이 비어있는 경우도 체크하는 것이 안전합니다.
        if image and image.filename: 
            image_data = await image.read()
            
            # 실제 파일 데이터가 있을 때만 Custom Vision 호출
            if len(image_data) > 0:
                image_ai_res = await analyze_image_ai(image_data)
                uploaded_url = await upload_image_to_blob(image_data, image.filename, image.content_type)

        # 4. 판별 라벨 결정 (이미지 분석 결과가 있을 때만 적용)
        final_label = text_ai_res["label"]
        if image_ai_res["label"].lower() not in ["clean", "no_image", "error"]:
            if image_ai_res["probability"] > 0.6:
                final_label = f"unsafe_image_{image_ai_res['label']}"

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
        print(f"🔥 서버 에러 상세: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "detail": str(e)}
        )

# --- [4. 조회 및 템플릿 로직] ---
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