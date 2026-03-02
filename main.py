from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from azure.storage.blob import BlobServiceClient

# 우리가 만든 파일들 불러오기
from database import get_db, engine
import models

app = FastAPI()

# --- [Azure Blob Storage 설정] ---
# ⭐️ 메모장에 적어둔 '연결 문자열'을 여기에 붙여넣으세요!
AZURE_STORAGE_CONNECTION_STRING = "***REMOVED***"
CONTAINER_NAME = "images" 

# 클라이언트 초기화
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
except Exception as e:
    print(f"⚠️ Azure Storage 연결 실패: {e}")

# 이미지 업로드 공통 함수
async def upload_image_to_blob(file: UploadFile):
    if not file:
        return None
    
    # 고유한 파일명 생성 (확장자 유지)
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
    
    contents = await file.read()
    blob_client.upload_blob(contents)
    
    return blob_client.url  # 저장된 이미지의 공개 URL 반환
# ---------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
                "image_url": c.image_url, # ⭐️ 댓글 이미지 URL 추가
                "username": c.author.username if c.author else "익명",
                "role": c.author.role if c.author else "user"
            })

        post_data = {
            "id": post.id,
            "body": post.body,
            "content": post.body,
            "username": post.author.username if post.author else "익명",
            "role": post.author.role if post.author else "user",
            "created_at": post.created_at.isoformat() if post.created_at else None,
            "comments": comment_list
        }
        result.append(post_data)
    return result

# 게시글 생성 (기존 유지)
class PostCreate(BaseModel):
    content: str
    user: str

@app.post("/posts")
async def create_post(item: PostCreate, db: Session = Depends(get_db)):
    actual_user_id = 4 
    try:
        new_post = models.Post(
            title="테스트 제목",
            body=item.content,
            user_id=actual_user_id,
            status="active"
        )
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        return {"status": "success", "post": new_post}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ⭐️ 댓글 생성 (이미지 업로드를 위해 FormData 방식으로 변경)
@app.post("/comments")
async def create_comment(
    content: str = Form(...),          # JSON이 아닌 Form으로 받음
    post_id: int = Form(...),          # JSON이 아닌 Form으로 받음
    image: Optional[UploadFile] = File(None), # 이미지 파일 (선택)
    db: Session = Depends(get_db)
):
    test_user_id = 6
    
    # 1. 이미지 있다면 Azure에 업로드
    uploaded_url = None
    if image:
        try:
            uploaded_url = await upload_image_to_blob(image)
        except Exception as e:
            print(f"❌ 이미지 업로드 실패: {e}")
            # 업로드 실패해도 댓글은 달릴 수 있게 하거나, 에러를 던질 수 있습니다.

    # 2. DB 저장 (image_url 컬럼에 주소 저장)
    new_comment = models.Comment(
        post_id=post_id,
        user_id=test_user_id,
        content=content,
        image_url=uploaded_url, # ⭐️ Blob 주소 저장
        toxicity_score=0.0,
        label="safe"
    )
    
    try:
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        return {"status": "success", "comment": new_comment}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))