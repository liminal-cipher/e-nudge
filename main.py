from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
from azure.storage.blob import BlobServiceClient

from database import get_db, engine
import models

app = FastAPI()

# --- [1. Azure Blob Storage 설정] ---
AZURE_STORAGE_CONNECTION_STRING = "여기에_실제_연결_문자열_입력"
CONTAINER_NAME = "images" 

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Storage 연결 설정 실패: {e}")

async def upload_image_to_blob(file: UploadFile):
    # 파일이 없거나 이름이 비어있으면 중단
    if not file or not file.filename:
        print("❌ 업로드 시도 실패: 파일이 없습니다.")
        return None
    
    try:
        # 고유 파일명 생성
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg" # 확장자 없을 경우 기본값
        filename = f"{uuid.uuid4()}{ext}"
        
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        
        # 파일 데이터 읽기
        contents = await file.read()
        
        # Azure에 실제 업로드 (데이터 타입 명시)
        blob_client.upload_blob(contents, overwrite=True)
        
        # ⭐️ 업로드된 실제 URL 생성 및 반환
        image_url = blob_client.url
        print(f"✅ Azure 업로드 성공! URL: {image_url}")
        return image_url
    except Exception as e:
        print(f"❌ Azure 업로드 중 서버 에러 발생: {str(e)}")
        return None

# --- [2. 게시글 로직] ---
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
                "id": c.id,
                "content": c.content,
                "image_url": c.image_url, # DB에서 가져온 URL
                "username": c.author.username if c.author else "익명",
                "role": c.author.role if c.author else "user"
            })
        result.append({
            "id": post.id,
            "body": post.body,
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

# --- [3. 댓글 로직: 텍스트 + 이미지 통합] ---
@app.post("/comments")
async def create_comment(
    content: str = Form(...),
    post_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    print(f"📩 댓글 요청 수신: post_id={post_id}, content={content}")
    
    # 1. 이미지 업로드 먼저 진행
    uploaded_url = None
    if image and image.filename:
        uploaded_url = await upload_image_to_blob(image)
        print(f"🔗 DB에 저장될 URL: {uploaded_url}")

    # 2. DB 저장
    try:
        new_comment = models.Comment(
            post_id=post_id,
            user_id=6,
            content=content,
            image_url=uploaded_url, # 이 변수가 None이면 DB에도 NULL로 들어감
            toxicity_score=0.0,
            label="safe"
        )
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        print("💾 DB 저장 완료!")
        return {"status": "success", "image_url": uploaded_url}
    except Exception as e:
        db.rollback()
        print(f"❌ DB 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="데이터베이스 저장 오류")

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})