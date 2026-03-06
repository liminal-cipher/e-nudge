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
import time
from database import get_db, engine
import models
from datetime import datetime
# DB 테이블 생성
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# 정적 파일 설정 (CSS, JS 등을 static 폴더에 넣을 경우 필요)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- [0. AI 모델 설정값] ---
CUSTOM_VISION_URL = "https://gdscs-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/720991f1-25e4-4d32-968d-0e00abbb1166/classify/iterations/Iteration5/image"
CUSTOM_VISION_KEY = "***REMOVED***"

app.mount("/static", StaticFiles(directory="static"), name="static")

# 모델 로드
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
    try:
        tokens = kiwi.tokenize(text, normalize_coda=True)
        return ' '.join(t.form for t in tokens if t.tag in KEEP_TAGS)
    except:
        return ""

async def analyze_text_ai(text: str):
    if not text: 
        return {"label": "none", "score": 0.0}
    try:
        def predict_sync():
            cleaned = clean_text(text)
            tokenized = tokenize(cleaned)
            if not tokenized: return None
            vector = tfidf_vectorizer.transform([tokenized])
            probs = lr_model.predict_proba(vector)[0]
            print(f"hate: {probs[0]}, none: {probs[1]}, offensive: {probs[2]}")  # 추가, 결과값 확인용
            return probs
        
        probs = await run_in_threadpool(predict_sync)
        if probs is None: return {"label": "none", "score": 0.0}

        hate_prob = probs[0]      # ['hate', 'none', 'offensive'] 순서
        offensive_prob = probs[2]

        toxicity = offensive_prob * 0.5 + hate_prob * 1.0

        if toxicity >= 0.55:
            label = "hate"
        elif toxicity >= 0.44:
            label = "offensive"
        else:
            label = "none"

        return {"label": label, "score": float(toxicity)}
    except Exception as e:
        print(f"❌ [TEXT AI] 분석 오류: {e}")
        return {"label": "error", "score": 0.0}
async def analyze_image_ai(image_bytes: bytes):
    if not image_bytes:
        return {"label": "no_image", "probability": 0.0}
    
    try:
        headers = {"Prediction-Key": CUSTOM_VISION_KEY, "Content-Type": "application/octet-stream"}
        async with httpx.AsyncClient() as client:
            response = await client.post(CUSTOM_VISION_URL, content=image_bytes, headers=headers, timeout=10.0)
            
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            
            if predictions:
                # 가장 확률이 높은 결과 하나만 가져옴
                top = max(predictions, key=lambda x: x['probability'])
                prob = float(top['probability'])
                
                # [수정] 90% 이상일 때만 'hate', 아니면 무조건 'none'
                # tagName이 무엇이든 '불쾌한 이미지' 모델이므로 확률만 체크
                if prob >= 0.9:
                    return {"label": "hate", "probability": prob}
                else:
                    return {"label": "none", "probability": prob}
            
        return {"label": "none", "probability": 0.0}
    except Exception as e:
        print(f"❌ [IMAGE AI] 예외: {e}")
        return {"label": "none", "probability": 0.0}

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

        # 1. 텍스트 분석 및 시간 측정 (로그: complementnb_v4)
        if content and content.strip():
            t_start = time.time() # 텍스트 타이머 시작
            text_ai_res = await analyze_text_ai(content)
            t_duration = round(time.time() - t_start, 4)
            
            # 텍스트 로그 개별 저장
            db.add(models.MLModel(model_version='complementnb_v4', inference_time=t_duration))

        # 2. 이미지 처리 및 시간 측정 (로그: CV)
        if image and image.filename and len(image.filename.strip()) > 0:
            image_data = await image.read()
            if image_data:
                i_start = time.time() # 이미지 타이머 시작
                image_task = analyze_image_ai(image_data)
                upload_task = upload_image_to_blob(image_data, image.filename, image.content_type)
                image_ai_res, uploaded_url = await asyncio.gather(image_task, upload_task)
                
                i_duration = round(time.time() - i_start, 4)
                
                # 이미지 로그 개별 저장
                db.add(models.MLModel(model_version='CV', inference_time=i_duration))

        # --- 라벨 가중치 판별 로직 (기존과 동일) ---
        label_weights = {"hate": 2, "offensive": 1, "none": 0, "clean": 0, "neutral": 0, "unknown": 0, "error": 0}
        text_label = text_ai_res.get("label", "none").lower()
        img_label = image_ai_res.get("label", "none").lower()

        final_weight = max(label_weights.get(text_label, 0), label_weights.get(img_label, 0))
        weight_to_label = {2: "hate", 1: "offensive", 0: "none"}
        final_label = weight_to_label[final_weight]
        
        raw_score = max(text_ai_res["score"], image_ai_res.get("probability", 0.0))
        final_score = round(float(raw_score), 2)

        # 3. Hate(2점) 등급이면 차단 (ML 로그 저장을 위해 커밋만 추가)
        if final_weight == 2:
            db.commit() 
            print(f"🚫 [차단] 유해 콘텐츠 감지 (Text: {text_label}, Image: {img_label})")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "blocked",
                    "message": "유해한 내용이 포함되어 등록이 차단되었습니다.",
                    "ai_result": {
                        "final": {"label": final_label, "score": final_score},
                        "detail": {"text": text_label, "image": img_label}
                    }
                }
            )

        # 4. 일반 저장 로직 (기존과 동일)
        new_comment = models.Comment(
            post_id=post_id,
            user_id=8, 
            content=content if content else "",
            image_url=uploaded_url,
            toxicity_score=float(final_score),
            label=final_label
        )
        db.add(new_comment)
        db.flush() 

        # --- [AdminLog 기록] ---
        new_admin_log = models.AdminLog(comments_id=new_comment.id)
        db.add(new_admin_log)
        
        db.commit()
        db.refresh(new_comment)
        
        print(f"✅ DB 저장 완료: ID {new_comment.id} (Label: {final_label})")

        return {
            "status": "success", 
            "id": new_comment.id,
            "ai_result": {
                "text": text_ai_res, 
                "image": image_ai_res,
                "final": {"label": final_label, "score": final_score}
            }
        }

    except Exception as e:
        if db: db.rollback()
        print(f"🔥 에러 발생: {e}")
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
                "label": c.label, 
                "toxicity_score": c.toxicity_score  # <--- "score"에서 "toxicity_score"로 변경
            } for c in post.comments]
        })
    return result

# --- [4. 페이지 렌더링 및 삭제 관리] ---

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request, db: Session = Depends(get_db)):
    # ID가 8인 유저의 동의 여부 확인
    user = db.query(models.User).filter(models.User.id == 8).first()
    
    # 유저가 없거나 false면 기본값 False
    is_agreed = user.is_agreed if user else False
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "is_agreed": is_agreed  # 템플릿으로 상태 전달
    })

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/demo", response_class=HTMLResponse)
async def read_demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

@app.delete("/comments/{comment_id}")
async def delete_comment(comment_id: int, db: Session = Depends(get_db)):
    try:
        comment = db.query(models.Comment).filter(models.Comment.id == comment_id).first()
        if not comment:
            raise HTTPException(status_code=404, detail="댓글을 찾을 수 없습니다.")
        
        db.delete(comment)
        db.commit()
        return {"status": "success", "message": f"댓글 {comment_id}이(가) 삭제되었습니다."}
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"detail": str(e)})
    
@app.post("/users/{user_id}/agreement")
async def update_user_agreement(
    user_id: int, 
    data: dict, 
    db: Session = Depends(get_db)
):
    """
    프론트엔드에서 전달된 유저의 개인정보 수집 동의 여부를 DB에 기록합니다.
    """
    try:
        is_agreed = data.get("is_agreed", False)
        
        # 유저 정보 조회
        user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if not user:
            # 실습용 환경이므로 유저가 없을 경우 생성하거나 에러를 반환
            raise HTTPException(status_code=404, detail="User not found")

        # 동의 여부 및 동의 시간 업데이트
        user.is_agreed = is_agreed
        user.agreed_at = datetime.now() if is_agreed else None
        
        db.commit()
        
        print(f"👤 [User {user_id}] 개인정보 동의 업데이트: {is_agreed}")
        return {"status": "success", "user_id": user_id, "is_agreed": is_agreed}

    except Exception as e:
        db.rollback()
        print(f"❌ 동의 정보 업데이트 실패: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/users/{user_id}")
async def get_user_info(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        # 유저가 없으면 기본값 리턴 (에러 방지)
        return {"id": user_id, "is_agreed": False}
    
    # DB의 True/False가 JSON의 true/false로 정확히 변환되어 나갑니다.
    return {"id": user.id, "is_agreed": user.is_agreed}