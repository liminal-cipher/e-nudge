from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import httpx  # 비동기 HTTP 통신을 위해 필수
from azure.storage.blob import BlobServiceClient, ContentSettings

from database import get_db, engine
import models

app = FastAPI()

# --- [0. AI 모델 설정값] ---
# Azure Portal의 Custom Vision -> Prediction 탭에서 확인한 값을 넣어주세요.
CUSTOM_VISION_URL = "https://gdscs.cognitiveservices.azure.com/"
CUSTOM_VISION_KEY = "CoaHfafsWECf6ary5nnHeFQV2Ff6P7mkDBbxyL4QiETeYSbCrRkZJQQJ99CBACYeBjFXJ3w3AAAJACOG96Zf"

# 텍스트 모델은 나중에 해결되면 여기에 정보를 넣으세요.
TEXT_MODEL_URL = "실제_ML_Designer_URL"
TEXT_MODEL_KEY = "실제_ML_Designer_KEY"

# --- [1. Azure Blob Storage 설정] ---
AZURE_STORAGE_CONNECTION_STRING = "***REMOVED***"
CONTAINER_NAME = "images" 

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Storage 연결 설정 실패: {e}")

# --- [AI 분석 보조 함수들] ---

async def analyze_text_ai(text: str):
    """ML Designer 텍스트 모델 호출 (현재 시뮬레이션 모드)"""
    if not text: return {"label": "safe", "score": 0.0}
    # 텍스트 모델 이슈 해결 전까지는 안전한 것으로 간주
    print(f"🔍 [TEXT AI] 시뮬레이션 중: '{text[:10]}...'")
    return {"label": "safe", "score": 0.05}

async def analyze_image_ai(image_bytes: bytes):
    """Custom Vision 이미지 모델 실제 호출"""
    if not image_bytes: 
        return {"label": "clean", "probability": 0.0}
    
    try:
        headers = {
            "Prediction-Key": CUSTOM_VISION_KEY,
            "Content-Type": "application/octet-stream"
        }
        
        async with httpx.AsyncClient() as client:
            # 7초 타임아웃 설정 (이미지 분석은 시간이 조금 걸릴 수 있음)
            response = await client.post(
                CUSTOM_VISION_URL, 
                content=image_bytes, 
                headers=headers, 
                timeout=7.0
            )
            
        if response.status_code == 200:
            result = response.json()
            # 가장 확률이 높은 예측값 가져오기
            top_prediction = result['predictions'][0]
            tag_name = top_prediction['tagName']
            probability = top_prediction['probability']
            
            print(f"✨ [IMAGE AI] 분석 완료: {tag_name} ({probability:.2%})")
            return {"label": tag_name, "probability": probability}
        else:
            print(f"⚠️ [IMAGE AI] API 응답 에러: {response.status_code}")
            return {"label": "error", "probability": 0.0}
            
    except Exception as e:
        print(f"❌ [IMAGE AI] 호출 중 예외 발생: {e}")
        return {"label": "error", "probability": 0.0}

async def upload_image_to_blob(file: UploadFile):
    if not file or not file.filename:
        return None
    try:
        await file.seek(0)
        contents = await file.read()
        content_type = file.content_type
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        blob_client.upload_blob(
            contents, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 실패: {e}")
        return None

# --- [2. 게시글 로직] --- (기존 유지)
class PostCreate(BaseModel):
    content: str
    user: Optional[str] = "익명"

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

@app.post("/posts")
async def create_post(item: PostCreate, db: Session = Depends(get_db)):
    try:
        new_post = models.Post(body=item.content, user_id=4, status="active")
        db.add(new_post)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# --- [3. 댓글 로직: Custom Vision 우선 적용] ---
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

        # AI 결과 초기화
        text_ai_res = {"label": "safe", "score": 0.0}
        image_ai_res = {"label": "clean", "probability": 0.0}

        # 1. 텍스트 분석 (시뮬레이션 모드)
        if content:
            text_ai_res = await analyze_text_ai(content)

        # 2. 이미지 분석 (실제 Custom Vision 호출)
        image_data = None
        if image:
            image_data = await image.read()
            image_ai_res = await analyze_image_ai(image_data)

        # 3. Azure Blob 업로드 (분석 후 진행)
        uploaded_url = None
        if image:
            await image.seek(0)
            uploaded_url = await upload_image_to_blob(image)
        
        # 4. 판별 라벨 결정 (Custom Vision 결과 우선 적용)
        # 만약 'clean'이 아닌 다른 태그(예: Unsafe)가 나오고 확률이 60% 이상이면 위험으로 간주
        final_label = text_ai_res["label"] 
        if image_ai_res and image_ai_res["label"].lower() != "clean":
            if image_ai_res["probability"] > 0.6:
                final_label = f"unsafe_image_{image_ai_res['label']}"

        # 5. DB 저장
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
                "image": image_ai_res  # 👈 여기에 커스텀 비전의 label과 probability가 들어있습니다.
        }
}
        
    except Exception as e:
        db.rollback()
        print(f"❌ 댓글 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="서버 처리 중 오류가 발생했습니다.")

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})