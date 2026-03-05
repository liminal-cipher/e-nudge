from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import httpx
import re
import emoji
import joblib
import asyncio
from kiwipiepy import Kiwi
from azure.storage.blob import BlobServiceClient, ContentSettings
from fastapi.concurrency import run_in_threadpool

from database import get_db, engine
import models

# DB 테이블 생성
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- [0. AI 모델 설정값] ---
CUSTOM_VISION_URL = "https://gdscs-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/720991f1-25e4-4d32-968d-0e00abbb1166/classify/iterations/Iteration5/image"
CUSTOM_VISION_KEY = "***REMOVED***"

# 모델 로드 (경로 확인 필요)
try:
    lr_model = joblib.load("lr_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    kiwi = Kiwi(typos="basic")
    print("✅ [TEXT AI] 모델 및 벡터라이저 로드 성공")
except Exception as e:
    print(f"❌ [TEXT AI] 로드 실패: {e}")

KEEP_TAGS = {'NNG', 'NNP', 'NP', 'VV', 'VA', 'VV-I', 'VA-I', 'VV-R', 'VA-R', 'MAG', 'MM', 'SW', 'IC'}

# --- [1. Azure Blob Storage 설정] ---
AZURE_STORAGE_CONNECTION_STRING = "***REMOVED***"
CONTAINER_NAME = "images" 

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Storage 연결 설정 실패: {e}")

# --- [2. AI 분석 보조 함수들] ---

def clean_text(text):
    if not isinstance(text, str): return ""
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    if not text: return ""
    tokens = kiwi.tokenize(text, normalize_coda=True)
    return ' '.join(t.form for t in tokens if t.tag in KEEP_TAGS)

async def analyze_text_ai(text: str):
    if not text: 
        return {"label": "none", "score": 0.0}
    try:
        def predict_sync():
            cleaned = clean_text(text)
            tokenized = tokenize(cleaned)
            if not tokenized: return None
            vector = tfidf_vectorizer.transform([tokenized])
            # 확률 추출 (none, offensive, hate 순서 가정)
            return lr_model.predict_proba(vector)[0]

        probs = await run_in_threadpool(predict_sync)
        if probs is None: return {"label": "none", "score": 0.0}

        labels = ['none', 'offensive', 'hate']
        max_idx = probs.argmax()
        return {"label": labels[max_idx], "score": float(probs[max_idx])}
    except Exception as e:
        print(f"❌ [TEXT AI] 분석 오류: {e}")
        return {"label": "error", "score": 0.0}

async def analyze_image_ai(image_bytes: bytes):
    if not image_bytes: return {"label": "no_image", "probability": 0.0}
    try:
        headers = {"Prediction-Key": CUSTOM_VISION_KEY, "Content-Type": "application/octet-stream"}
        async with httpx.AsyncClient() as client:
            response = await client.post(CUSTOM_VISION_URL, content=image_bytes, headers=headers, timeout=10.0)
            
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            if predictions:
                top = max(predictions, key=lambda x: x['probability'])
                print(f"📸 [IMAGE AI] 분석 완료: {top['tagName']} ({top['probability']:.2%})")
                return {"label": top['tagName'], "probability": float(top['probability'])}
        
        return {"label": "unknown", "probability": 0.0}
    except Exception as e:
        print(f"❌ [IMAGE AI] 예외 발생: {e}")
        return {"label": "error", "probability": 0.0}

async def upload_image_to_blob(contents: bytes, filename: str, content_type: str):
    try:
        def upload_sync():
            ext = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4()}{ext}"
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=unique_filename)
            blob_client.upload_blob(contents, overwrite=True, content_settings=ContentSettings(content_type=content_type))
            return blob_client.url
        return await run_in_threadpool(upload_sync)
    except Exception as e:
        print(f"❌ [Storage] 업로드 실패: {e}")
        return None

# --- [3. API 엔드포인트] ---

@app.post("/comments")
async def create_comment(
    content: Optional[str] = Form(None),
    post_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        text_ai_res = {"label": "none", "score": 0.0}
        image_ai_res = {"label": "clean", "probability": 0.0}
        uploaded_url = None

        # 1. 텍스트 분석
        if content and content.strip():
            text_ai_res = await analyze_text_ai(content)

        # 2. 이미지 처리
        if image and image.filename:
            image_data = await image.read()
            if image_data:
                image_task = analyze_image_ai(image_data)
                upload_task = upload_image_to_blob(image_data, image.filename, image.content_type)
                image_ai_res, uploaded_url = await asyncio.gather(image_task, upload_task)

        # 3. 최종 판별 (이미지 우선순위 전략)
        final_label = text_ai_res["label"]
        img_label = image_ai_res["label"].lower()
        if img_label not in ["clean", "no_image", "error", "unknown"]:
            if image_ai_res["probability"] > 0.6:
                final_label = f"unsafe_image_{img_label}"

        # 4. DB 저장 (user_id는 현재 샘플로 6번 사용)
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
        
        return {"status": "success", "id": new_comment.id, "ai_result": {"text": text_ai_res, "image": image_ai_res}}

    except Exception as e:
        if db: db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/posts")
async def create_post(title: str = Form("제목 없음"), content: str = Form(...), db: Session = Depends(get_db)):
    try:
        new_post = models.Post(title=title, body=content, user_id=6, status="active")
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        return new_post
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/posts")
async def get_posts(db: Session = Depends(get_db)):
    posts = db.query(models.Post).order_by(models.Post.id.desc()).all()
    result = []
    for post in posts:
        result.append({
            "id": post.id, "body": post.body,
            "username": post.author.username if post.author else "익명",
            "role": post.author.role if post.author else "user",
            "created_at": post.created_at.isoformat() if post.created_at else None,
            "comments": [{
                "id": c.id, "content": c.content, "image_url": c.image_url,
                "username": c.author.username if c.author else "익명",
                "label": c.label, "score": c.toxicity_score
            } for c in post.comments]
        })
    return result

# --- [4. 페이지 렌더링] ---

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# 서버 실행 시: uvicorn main:app --reload