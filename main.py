from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
from azure.storage.blob import BlobServiceClient

# 우리가 만든 파일들 불러오기
from database import get_db, engine
import models

app = FastAPI()

# --- [1. Azure Blob Storage 설정] ---
# ⭐️ 실제 연결 문자열로 교체하세요 (깃허브 푸시 시 주의!)
AZURE_STORAGE_CONNECTION_STRING = "여기에_연결_문자열_입력"
CONTAINER_NAME = "images" 

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Storage 연결 실패: {e}")

async def upload_image_to_blob(file: UploadFile):
    if not file or not file.filename:
        return None
    
    try:
        # 고유 파일명 생성
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        
        # 파일 읽기 및 업로드
        contents = await file.read()
        blob_client.upload_blob(contents, overwrite=True)
        
        return blob_client.url
    except Exception as e:
        print(f"❌ 블롭 업로드 에러: {e}")
        return None

# --- [2. 게시글 로직] ---
class PostCreate(BaseModel):
    content: str
    user: Optional[str] = "익명" # 필수 값이 아님을 명시하여 에러 방지

@app.get("/posts")
async def get_posts(db: Session = Depends(get_db)):
    # 최신순 정렬
    posts = db.query(models.Post).order_by(models.Post.id.desc()).all()
    result = []
    for post in posts:
        comment_list = []
        for c in post.comments:
            comment_list.append({
                "id": c.id,
                "content": c.content,
                "image_url": c.image_url,
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
        # 실제 DB에 존재하는 user_id(4)를 사용
        new_post = models.Post(
            title="새 게시글",
            body=item.content, 
            user_id=4, 
            status="active"
        )
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        return {"status": "success", "post_id": new_post.id}
    except Exception as e:
        db.rollback()
        print(f"❌ 게시글 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- [3. 댓글 로직 (Form 데이터 방식)] ---
@app.post("/comments")
async def create_comment(
    content: Optional[str] = Form(None), # 👈 Form(...)에서 Form(None)으로 변경!
    post_id: int = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        # 이미지가 있거나 텍스트가 있거나 둘 중 하나는 있어야 저장 진행
        if not content and not image:
            raise HTTPException(status_code=400, detail="내용이나 이미지 중 하나는 필요합니다.")

        # 이미지 업로드 처리
        uploaded_url = await upload_image_to_blob(image) if image else None
        
        new_comment = models.Comment(
            post_id=post_id,
            user_id=6,
            content=content if content else "", # 👈 내용이 없으면 빈 문자열 저장
            image_url=uploaded_url,
            toxicity_score=0.0,
            label="safe"
        )
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        return {"status": "success", "comment_id": new_comment.id}
    except Exception as e:
        db.rollback()
        print(f"❌ 댓글 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- [기본 경로 설정] ---
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})