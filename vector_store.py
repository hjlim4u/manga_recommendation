from typing import List, Optional
import logging
import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, Range
from domain import AgeRating
from data_source import MangaDataSource

# 로깅 설정
logger = logging.getLogger(__name__)

class QdrantMangaStore:
    """Qdrant 벡터 저장소를 사용한 만화 데이터 관리 클래스"""
    
    def __init__(self, collection_name: str = "manga_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Qdrant 클라이언트 초기화
        self.client = QdrantClient(host="localhost", port=6333)
        
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
    
    def is_collection_empty(self) -> bool:
        """컬렉션이 비어있는지 확인"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count == 0
        except Exception:
            return True
    
    def delete_collection(self):
        """컬렉션 삭제하고 새로 생성"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"컬렉션 '{self.collection_name}' 삭제됨")
        except Exception:
            pass
        
        self._create_collection_if_not_exists()
        # LangChain 래퍼 재초기화
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
    
    def load_and_index_from_source(self, data_source: MangaDataSource, source_batch_size: int = 1000, index_batch_size: int = 100, force_reindex: bool = False):
        """데이터 소스에서 만화 데이터를 스트리밍으로 로드하고 벡터 DB에 인덱싱 (메모리 효율적)"""
        
        # 데이터가 이미 있고 강제 재인덱싱이 아니면 건너뛰기
        if not force_reindex and not self.is_collection_empty():
            logger.info("✅ 컬렉션에 이미 데이터가 있습니다. 인덱싱을 건너뜁니다.")
            return
        
        total_count = data_source.get_total_count()
        logger.info(f"Starting streaming indexing from data source (Total: {total_count} records)")
        
        processed_count = 0
        batch_count = 0
        
        try:
            # 배치 스트리밍으로 데이터 처리
            for manga_batch in data_source.load_manga_data_batches(source_batch_size):
                batch_count += 1
                
                # Document 객체로 변환
                documents = data_source.create_documents(manga_batch)
                logger.info(f"Batch {batch_count}: Created {len(documents)} documents")
                
                # 벡터 DB에 인덱싱
                self.index_manga_batch(documents, index_batch_size)
                
                processed_count += len(documents)
                logger.info(f"Progress: {processed_count}/{total_count} records indexed ({processed_count/total_count*100:.1f}%)")
                
                # 메모리 절약: 배치 처리 후 즉시 해제
                del manga_batch, documents
            
            logger.info(f"Streaming indexing completed successfully: {processed_count} records in {batch_count} batches")
            
        except Exception as e:
            logger.error(f"Streaming indexing failed: {e}")
            raise
    
    def index_manga_batch(self, documents: List[Document], batch_size: int = 100):
        """대량의 만화 데이터를 배치로 인덱싱"""
        total = len(documents)
        logger.info(f"Starting batch indexing of {total} documents")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                # 연령 등급 값 추가
                age_rating = AgeRating.from_option(
                    doc.metadata.get('age_grad_cd_nm', '전체연령')
                )
                doc.metadata['age_rating_value'] = age_rating.value
                
                # ID 정수 변환
                if 'id' in doc.metadata:
                    doc.metadata['manga_id'] = int(doc.metadata['id'])
            
            self.vector_store.add_documents(batch)
            logger.info(f"Indexed {min(i + batch_size, total)}/{total} documents")
    
    def find_manga_by_title(self, target_title: str) -> Optional[Document]:
        """🚀 간단한 제목 검색: 가장 유사한 만화 1개 반환"""
        try:
            logger.info(f"🔍 제목 검색: '{target_title}'")
            
            # 1단계: 정확 매칭 시도
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="metadata.title",
                        match=MatchValue(value=target_title)
                    )]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_result[0]:  # 정확 매칭 성공
                point = search_result[0][0]
                logger.info(f"✅ 정확 매칭: '{target_title}'")
                return Document(
                    page_content=point.payload['page_content'],
                    metadata=point.payload['metadata']
                )
            
            # 2단계: 벡터 유사도 검색으로 가장 유사한 만화 1개 찾기
            logger.info(f"정확 매칭 실패, 유사도 검색 시도...")
            
            # 제목을 임베딩으로 변환
            title_embedding = self.embeddings.embed_query(target_title)
            
            # 벡터 검색으로 가장 유사한 만화 찾기
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=title_embedding,
                limit=1,  # 가장 유사한 1개만
                with_payload=True,
                with_vectors=False,
                score_threshold=0.1  # 매우 낮은 임계값 (거의 모든 결과 허용)
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
    
    def search_similar_manga_by_centroid(self, favorite_manga_docs: List[Document], 
                                       preferred_genres: List[str], max_age_rating: int, 
                                       limit: int = 30) -> List[Document]:
        """중심점 임베딩을 사용한 유사 만화 검색"""
        try:
            if not favorite_manga_docs:
                logger.warning("좋아하는 만화가 없어서 중심점 검색 불가능")
                return []
            
            exclude_ids = [doc.metadata['manga_id'] for doc in favorite_manga_docs]
            logger.info(f"중심점 검색 - 좋아하는 만화 {len(favorite_manga_docs)}개, 제외 ID: {exclude_ids}")
            
            # 중심점 임베딩 생성
            embeddings = []
            for doc in favorite_manga_docs:
                embedding = self.embeddings.embed_query(doc.page_content)
                embeddings.append(embedding)
            
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            
            # 검색 필터 구성
            search_filter = self._build_search_filter(preferred_genres, max_age_rating, exclude_ids)
            
            # 벡터 검색 실행
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=centroid.tolist(),
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"중심점 검색 결과: {len(search_result.points)}개")
            
            # Document 변환
            documents = []
            for point in search_result.points:
                metadata = point.payload.get('metadata', {}).copy()
                metadata['similarity_score'] = point.score
                
                doc = Document(
                    page_content=point.payload.get('page_content', ''),
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"중심점 검색 오류: {e}")
            return []
    
    def search_similar_manga_by_individual(self, favorite_manga_docs: List[Document], 
                                         preferred_genres: List[str], max_age_rating: int, 
                                         limit_per_manga: int = 15) -> List[Document]:
        """개별 만화 검색 후 병합"""
        try:
            if not favorite_manga_docs:
                logger.warning("좋아하는 만화가 없어서 개별 검색 불가능")
                return []
            
            exclude_ids = [doc.metadata['manga_id'] for doc in favorite_manga_docs]
            logger.info(f"개별 검색 - 좋아하는 만화 {len(favorite_manga_docs)}개")
            
            all_results = {}
            
            for target_doc in favorite_manga_docs:
                # 개별 임베딩으로 검색
                embedding = self.embeddings.embed_query(target_doc.page_content)
                
                # 검색 필터 구성
                search_filter = self._build_search_filter(preferred_genres, max_age_rating, exclude_ids)
                
                # 벡터 검색 실행
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    query_filter=search_filter,
                    limit=limit_per_manga
                )
                
                logger.info(f"'{target_doc.metadata['title']}' 검색 결과: {len(search_result.points)}개")
                
                # 점수 누적
                for point in search_result.points:
                    manga_id = point.payload['metadata']['manga_id']
                    if manga_id not in all_results:
                        all_results[manga_id] = {
                            'point': point,
                            'total_score': 0,
                            'count': 0
                        }
                    all_results[manga_id]['total_score'] += point.score
                    all_results[manga_id]['count'] += 1
            
            # 평균 점수로 정렬
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: x['total_score'] / x['count'],
                reverse=True
            )
            
            logger.info(f"개별 검색 병합 결과: {len(sorted_results)}개")
            
            # Document 변환
            documents = []
            for item in sorted_results[:30]:  # 상위 30개만
                point = item['point']
                metadata = point.payload.get('metadata', {}).copy()
                metadata['similarity_score'] = item['total_score'] / item['count']
                
                doc = Document(
                    page_content=point.payload.get('page_content', ''),
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"개별 검색 오류: {e}")
            return []
    
    def _build_search_filter(self, preferred_genres: List[str], max_age_rating: int, 
                           exclude_ids: List[int]) -> Filter:
        """검색 필터 구성"""
        # 필수 조건
        must_conditions = [
            # FieldCondition(
            #     key="metadata.main_genre_cd_nm",
            #     match=MatchAny(any=preferred_genres)
            # ),
            FieldCondition(
                key="metadata.age_rating_value",
                range=Range(lte=max_age_rating)
            )
        ]
        
        # 제외 조건
        must_not_conditions = []
        if exclude_ids:
            for exclude_id in exclude_ids:
                must_not_conditions.append(
                    FieldCondition(
                        key="metadata.manga_id",
                        match=MatchValue(value=exclude_id),
                    )
                )
        
        return Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None
        )
    
    def debug_vector_db_contents(self, limit: int = 10):
        """🔍 벡터 DB 내용 디버깅 - 상세 정보 포함"""
        try:
            # 벡터 DB에서 샘플 가져오기 (scroll 사용)
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"🔍 벡터 DB 샘플 제목들 ({len(search_result[0])}개):")
            for i, point in enumerate(search_result[0]):
                metadata = point.payload.get('metadata', {})
                title = metadata.get('title', 'N/A')
                full_title = metadata.get('full_title', 'N/A') 
                manga_id = metadata.get('manga_id', 'N/A')
                logger.info(f"  {i+1}. ID:{manga_id} | title:'{title}' | full_title:'{full_title}'")
                
        except Exception as e:
            logger.error(f"벡터 DB 디버깅 실패: {e}") 