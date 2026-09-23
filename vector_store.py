from typing import Any, Dict, List, Optional
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType
from data_source import create_documents
import os
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

class QdrantMangaStore:
    """Qdrant 벡터 저장소를 사용한 만화 데이터 관리 클래스"""

    def __init__(self, collection_name: str = "manga_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # 컬렉션 생성
        self._create_collection_if_not_exists()

        # LangChain Qdrant 래퍼
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def _create_collection_if_not_exists(self):
        """컬렉션이 없으면 생성"""
        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=3072,  # text-embedding-3-large 차원
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

            # 제목 필드들에 대한 keyword 인덱스 생성
            title_fields = ['metadata.title', 'metadata.title_english', 'metadata.title_japanese', "metadata.manga_id"]
            for field in title_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_type=PayloadSchemaType.KEYWORD if field != "metadata.manga_id" else PayloadSchemaType.INTEGER
                    )
                    logger.info(f"Created keyword index for: {field}")
                except Exception as e:
                    logger.warning(f"Failed to create index for {field}: {e}")

    def is_collection_empty(self) -> bool:
        """컬렉션이 비어있는지 확인"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count == 0
        except Exception:
            return True

    def load_and_index_from_source(self, manga_data: List[Dict[str, Any]], index_batch_size: int = 100, force_reindex: bool = False):
        """정규화된 만화 데이터를 Document로 변환하고 벡터 DB에 인덱싱"""

        # 데이터가 이미 있고 강제 재인덱싱이 아니면 건너뛰기
        if not force_reindex and not self.is_collection_empty():
            logger.info("✅ 컬렉션에 이미 데이터가 있습니다. 인덱싱을 건너뜁니다.")
            return

        documents = create_documents(manga_data)
        logger.info(f"Starting indexing of {len(documents)} documents")
        self.index_manga_batch(documents, index_batch_size)
        logger.info(f"Indexing completed successfully: {len(documents)} documents")

    def index_manga_batch(self, documents: List[Document], batch_size: int = 100):
        """대량의 만화 데이터를 배치로 인덱싱"""
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            logger.info(f"Indexed {min(i + batch_size, total)}/{total} documents")

    def find_manga_by_title(self, target_title: str) -> Optional[Document]:
        """🚀 제목 검색 (새로운 형식 지원: title, title_english, title_japanese)"""
        try:
            logger.info(f"🔍 제목 검색: '{target_title}'")

            # 1단계: 여러 제목 필드에서 정확 매칭 시도
            title_fields = ['metadata.title', 'metadata.title_english', 'metadata.title_japanese']

            for field in title_fields:
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key=field,
                            match=MatchValue(value=target_title)
                        )]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

                if search_result[0]:  # 정확 매칭 성공
                    point = search_result[0][0]
                    matched_title = point.payload['metadata'].get(field.split('.')[-1], '')
                    logger.info(f"✅ 정확 매칭 ({field}): '{target_title}' → '{matched_title}'")
                    return Document(
                        page_content=point.payload['page_content'],
                        metadata=point.payload['metadata']
                    )

            # 2단계: 벡터 유사도 검색
            logger.info(f"정확 매칭 실패, 유사도 검색 시도...")

            # 제목을 임베딩으로 변환
            title_embedding = self.embeddings.embed_query(target_title)

            # 벡터 검색으로 가장 유사한 만화 찾기
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=title_embedding,
                limit=1,
                with_payload=True,
                with_vectors=False,
                score_threshold=0.1
            )

            if search_results.points:
                point = search_results.points[0]
                db_title = point.payload['metadata']['title']
                logger.info(f"✅ 유사도 검색: '{target_title}' → '{db_title}' (점수: {point.score:.3f})")

                return Document(
                    page_content=point.payload['page_content'],
                    metadata=point.payload['metadata']
                )

            logger.warning(f"❌ 검색 실패: '{target_title}'")
            return None

        except Exception as e:
            logger.error(f"제목 검색 오류: {e}")
            return None

    def search_similar_manga(self, favorite_manga_docs: List[Document], limit: int = 30) -> List[Document]:
        """좋아하는 만화와 유사한 만화 검색 (중심점 임베딩)"""
        if not favorite_manga_docs:
            logger.warning("좋아하는 만화가 없어서 검색 불가능")
            return []

        try:
            exclude_ids = [doc.metadata['manga_id'] for doc in favorite_manga_docs]
            logger.info(f"유사 만화 검색 - 좋아하는 만화 {len(favorite_manga_docs)}개, 제외 ID: {exclude_ids}")

            query_text = " ".join(doc.page_content for doc in favorite_manga_docs)
            query_vector = self.embeddings.embed_query(query_text)

            search_filter = Filter(must_not=[
                FieldCondition(key="metadata.manga_id", match=MatchValue(value=eid))
                for eid in exclude_ids
            ])

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            logger.info(f"검색 결과: {len(search_result.points)}개")

            documents = []
            for point in search_result.points:
                metadata = point.payload.get('metadata', {}).copy()
                metadata['similarity_score'] = point.score

                documents.append(Document(
                    page_content=point.payload.get('page_content', ''),
                    metadata=metadata
                ))

            return documents

        except Exception as e:
            logger.error(f"유사 만화 검색 오류: {e}")
            return []
