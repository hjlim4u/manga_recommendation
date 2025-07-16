from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Generator
import pandas as pd
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MangaDataSource(ABC):
    """만화 데이터를 제공하는 추상 인터페이스"""
    
    @abstractmethod
    def get_total_count(self) -> int:
        """전체 만화 데이터 개수 반환"""
        pass
    
    @abstractmethod
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """만화 데이터를 배치 단위로 스트리밍하여 제너레이터로 반환"""
        pass
    
    def load_manga_data(self) -> List[Dict[str, Any]]:
        """전체 데이터를 한 번에 로드 (테스트용, 소규모 데이터만)"""
        logger.warning("load_manga_data()는 테스트용입니다. 대용량 데이터에는 load_manga_data_batches()를 사용하세요.")
        all_data = []
        for batch in self.load_manga_data_batches():
            all_data.extend(batch)
            if len(all_data) > 10000:  # 안전장치: 1만개 초과 시 경고
                logger.warning(f"대용량 데이터 감지: {len(all_data)}개. 배치 처리를 권장합니다.")
                break
        return all_data
    
    def create_documents(self, manga_data: List[Dict[str, Any]]) -> List[Document]:
        """만화 데이터를 Document 객체로 변환"""
        documents = []
        
        for i, manga in enumerate(manga_data):
            # 검색용 텍스트 구성 (제목 + 장르 + 줄거리)
            content = f"""
            {manga.get('prdct_nm', '')}
            {manga.get('main_genre_cd_nm', '')}
            {manga.get('outline', '')}
            """.strip()
            
            # 메타데이터 구성
            metadata = {
                'manga_id': int(manga.get('id', 0)),
                'title': str(manga.get('prdct_nm', '')),
                'subtitle': str(manga.get('subtitl', '')),
                'full_title': str(manga.get('title', '')),
                'main_genre_cd_nm': str(manga.get('main_genre_cd_nm', '')),
                'age_grad_cd_nm': str(manga.get('age_grad_cd_nm', '전체연령')),
                'author': str(manga.get('pictr_writr_nm', '')),
                'outline': str(manga.get('outline', '')),
                'image_url': str(manga.get('image_download_url', ''))
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
            # 첫 번째 문서 디버깅 정보
            if i == 0:
                logger.info(f"첫 번째 문서 메타데이터: {metadata}")
                logger.info(f"첫 번째 문서 내용 길이: {len(content)}")
        
        return documents


class CSVMangaDataSource(MangaDataSource):
    """CSV 파일에서 만화 데이터를 로드하는 구현체 (테스트용)"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._total_count = 0
        self._df_processed = None
    
    def _load_and_preprocess(self):
        """CSV 로드 및 전처리 (한 번만 실행)"""
        if self._df_processed is not None:
            return
            
        try:
            logger.info(f"Loading and preprocessing CSV: {self.csv_path}")
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"CSV 로드 완료: {len(df)} rows")
            
            # CSV 데이터 구조 확인
            logger.info(f"CSV 컬럼들: {list(df.columns)}")
            
            # 중복 제거: prdct_nm 기준으로 중복 제거
            original_count = len(df)
            df_dedup = df.drop_duplicates(subset=['prdct_nm'], keep='first')
            removed_count = original_count - len(df_dedup)
            
            logger.info(f"중복 제거 완료: {original_count} → {len(df_dedup)} rows ({removed_count}개 중복 제거)")
            
            self._total_count = len(df_dedup)
            self._df_processed = df_dedup
            
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            raise
    
    def get_total_count(self) -> int:
        self._load_and_preprocess()
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """CSV 데이터를 배치 단위로 스트리밍"""
        self._load_and_preprocess()
        
        total_rows = len(self._df_processed)
        logger.info(f"Starting batch streaming: {total_rows} total rows, batch size: {batch_size}")
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = self._df_processed.iloc[start_idx:end_idx]
            batch_data = batch_df.to_dict('records')
            
            logger.info(f"Yielding batch {start_idx//batch_size + 1}: rows {start_idx}-{end_idx-1} ({len(batch_data)} records)")
            yield batch_data


class DatabaseMangaDataSource(MangaDataSource):
    """데이터베이스에서 만화 데이터를 로드하는 구현체 (상용)"""
    
    def __init__(self, db_config: Dict[str, Any], db_batch_size: int = 10000):
        self.db_config = db_config
        self.db_batch_size = db_batch_size
        self._total_count = 0
        self._connection = None
    
    def _get_connection(self):
        """DB 연결 생성 (실제 구현 필요)"""
        if self._connection is None:
            # TODO: 실제 DB 연결 구현
            # self._connection = create_connection(self.db_config)
            pass
        return self._connection
    
    def get_total_count(self) -> int:
        """총 레코드 수 조회"""
        if self._total_count == 0:
            try:
                # TODO: 실제 구현
                # connection = self._get_connection()
                # cursor = connection.cursor()
                # total_query = "SELECT COUNT(*) FROM manga_table WHERE status = 'active'"
                # self._total_count = cursor.execute(total_query).fetchone()[0]
                
                # 현재는 예시 값
                raise NotImplementedError("Database total count not implemented yet")
                
            except Exception as e:
                logger.error(f"총 개수 조회 실패: {e}")
                raise
        
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """데이터베이스에서 배치 단위로 스트리밍 (메모리 효율적)"""
        logger.info(f"Starting database batch streaming with batch size: {batch_size}")
        
        try:
            # TODO: 실제 DB 구현
            # connection = self._get_connection()
            # cursor = connection.cursor()
            # 
            # # 커서 기반 스트리밍 쿼리 (메모리 효율적)
            # base_query = """
            #     SELECT id, prdct_nm, subtitl, title, main_genre_cd_nm, 
            #            age_grad_cd_nm, pictr_writr_nm, outline, image_download_url
            #     FROM manga_table 
            #     WHERE status = 'active'
            #     ORDER BY id
            # """
            # 
            # # 서버 사이드 커서 사용 (PostgreSQL 예시)
            # cursor.execute(base_query)
            # 
            # batch_count = 0
            # while True:
            #     batch_rows = cursor.fetchmany(batch_size)
            #     if not batch_rows:
            #         break
            #     
            #     batch_data = [dict(row) for row in batch_rows]
            #     batch_count += 1
            #     
            #     logger.info(f"DB Batch {batch_count}: {len(batch_data)} records")
            #     yield batch_data
            # 
            # logger.info(f"Database streaming completed: {batch_count} batches")
            
            # 현재는 NotImplemented
            raise NotImplementedError("Database batch streaming not implemented yet")
            
        except Exception as e:
            logger.error(f"Database 배치 스트리밍 실패: {e}")
            raise


class MockDatabaseMangaDataSource(MangaDataSource):
    """모킹된 대용량 데이터베이스 (개발/테스트용)"""
    
    def __init__(self, record_count: int = 1000000):
        self.record_count = record_count
        self._total_count = record_count
        
        # 모킹 데이터 템플릿
        self.genres = ["로맨스/순정", "액션", "드라마", "판타지", "코미디", "스릴러", "BL", "백합", "학원", "성인"]
        self.age_ratings = ["전체연령", "12세 이상", "15세 이상", "18세 이상"]
        self.authors = ["김작가", "이작가", "박작가", "최작가", "정작가", "강작가", "조작가", "윤작가"]
    
    def get_total_count(self) -> int:
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """대용량 모킹 데이터를 배치 단위로 스트리밍 생성"""
        logger.info(f"Starting mock data streaming: {self.record_count} total records, batch size: {batch_size}")
        
        total_batches = (self.record_count + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_id = batch_num * batch_size + 1
            end_id = min((batch_num + 1) * batch_size, self.record_count)
            batch_data = []
            
            for i in range(start_id, end_id + 1):
                # 다양성을 위한 모킹 데이터 생성
                genre_idx = i % len(self.genres)
                rating_idx = i % len(self.age_ratings)
                author_idx = i % len(self.authors)
                
                mock_record = {
                    'id': i,
                    'prdct_nm': f"모킹 만화 {i:06d}",
                    'subtitl': f"부제목 {i}",
                    'title': f"[{self.genres[genre_idx]}] 모킹 만화 {i:06d} - 완결",
                    'main_genre_cd_nm': self.genres[genre_idx],
                    'age_grad_cd_nm': self.age_ratings[rating_idx],
                    'pictr_writr_nm': f"{self.authors[author_idx]}{i//1000}",
                    'outline': f"이것은 {i}번째 모킹 만화입니다. {self.genres[genre_idx]} 장르의 흥미진진한 스토리를 담고 있습니다.",
                    'image_download_url': f"https://example.com/manga/image_{i:06d}.jpg"
                }
                batch_data.append(mock_record)
            
            logger.info(f"Mock batch {batch_num + 1}/{total_batches}: records {start_id}-{end_id} ({len(batch_data)} records)")
            yield batch_data
        
        logger.info(f"Mock data streaming completed: {total_batches} batches, {self.record_count} total records") 