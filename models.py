from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, Text, NVARCHAR, String  # ⭐️ NVARCHAR 추가
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

# 1. 유저 테이블
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(NVARCHAR) # ⭐️ String 대신 NVARCHAR
    role = Column(NVARCHAR) 
    created_at = Column(DateTime, server_default=func.now())

    posts = relationship("Post", back_populates="author")
    comments = relationship("Comment", back_populates="author")

# 2. 게시글 테이블
class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(NVARCHAR) # ⭐️ NVARCHAR로 변경 (collation 필요 없음)
    body = Column(NVARCHAR)  # ⭐️ Text 대신 NVARCHAR(max)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(NVARCHAR)
    created_at = Column(DateTime, server_default=func.now())

    author = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post")

# 3. 댓글 테이블
class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(NVARCHAR) # ⭐️ 한글 댓글을 위해 NVARCHAR(max)
    image_url = Column(NVARCHAR, nullable=True)
    toxicity_score = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    label = Column(NVARCHAR(100))

    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")
    admin_logs = relationship("AdminLog", back_populates="comment")

# 4. 관리자 로그 테이블
class AdminLog(Base):
    __tablename__ = "admin_log"
    id = Column(Integer, primary_key=True, index=True)
    comments_id = Column(Integer, ForeignKey("comments.id"), nullable=False)

    comment = relationship("Comment", back_populates="admin_logs")

# 5. 모델 관리 테이블
class MLModel(Base):
    __tablename__ = "ml_model"
    model_version = Column(NVARCHAR, primary_key=True) 
    inference_time = Column(Float)
    created_at = Column(DateTime, server_default=func.now())