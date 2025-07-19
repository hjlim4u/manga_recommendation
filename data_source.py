from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Generator
import pandas as pd
import logging
import json
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
    
    def _normalize_json_field(self, value: Any) -> List[str]:
        """JSON 필드를 일관된 리스트 형식으로 정규화"""
        if not value:
            return []
        
        try:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, list):
                return value
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _extract_image_url(self, images_data: Any) -> str:
        """이미지 데이터에서 URL 추출"""
        if not images_data:
            return ""
        
        try:
            if isinstance(images_data, str):
                images = json.loads(images_data)
            else:
                images = images_data
            
            # JPG 우선, 없으면 WebP
            if isinstance(images, dict):
                if images.get('jpg', {}).get('image_url'):
                    return images['jpg']['image_url']
                elif images.get('webp', {}).get('image_url'):
                    return images['webp']['image_url']
        except (json.JSONDecodeError, TypeError):
            pass
        
        return ""
    
    def _extract_published_info(self, published_data: Any) -> tuple[str, str, str]:
        """출간 정보 추출 - from, to 날짜와 string 반환"""
        if not published_data:
            return "", "", ""
        
        try:
            if isinstance(published_data, str):
                pub_data = json.loads(published_data)
            elif isinstance(published_data, dict):
                pub_data = published_data
            else:
                return "", "", str(published_data)
            
            # from과 to 날짜 추출
            from_date = ""
            to_date = ""
            
            if 'from' in pub_data:
                from_date = pub_data['from']
                # ISO 형식에서 날짜만 추출 (예: "1994-12-05T00:00:00+00:00" -> "1994-12-05")
                if isinstance(from_date, str) and 'T' in from_date:
                    from_date = from_date.split('T')[0]
            
            if 'to' in pub_data:
                to_date = pub_data['to']
                # ISO 형식에서 날짜만 추출
                if isinstance(to_date, str) and 'T' in to_date:
                    to_date = to_date.split('T')[0]
            
            # 기존 string 필드도 유지 (호환성)
            string_date = pub_data.get('string', '')
            
            return from_date, to_date
            
        except (json.JSONDecodeError, TypeError):
            return "", "", str(published_data)
    
    def _normalize_manga_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """원시 레코드를 정규화된 형식으로 변환"""
        # 출간 정보 추출
        published_from, published_to = self._extract_published_info(raw_record.get('published'))
        
        return {
            # 기본 필드들 (문자열)
            'id': int(raw_record.get('id', 0)),
            'title': str(raw_record.get('title', '')),
            'title_english': str(raw_record.get('title_english', '')),
            'title_japanese': str(raw_record.get('title_japanese', '')),
            'status': str(raw_record.get('status', '')),
            'synopsis': str(raw_record.get('synopsis', '')),
            'background': str(raw_record.get('background', '')),
            'created_at': str(raw_record.get('created_at', '')),
            
            # JSON 배열 필드들 (항상 리스트로 정규화)
            'genres': self._normalize_json_field(raw_record.get('genres')),
            'themes': self._normalize_json_field(raw_record.get('themes')),
            'demographics': self._normalize_json_field(raw_record.get('demographics')),
            'authors': self._normalize_json_field(raw_record.get('authors')),
            
            # 복잡한 객체 처리
            'image_url': self._extract_image_url(raw_record.get('images')),
            'published_start': published_from,    # 새로운 필드
            'published_end': published_to,        # 새로운 필드
        }
    
    def create_documents(self, manga_data: List[Dict[str, Any]]) -> List[Document]:
        """정규화된 데이터로부터 Document 생성 (날짜 정보 포함)"""
        documents = []
        
        for i, manga in enumerate(manga_data):
            
            # 모든 content 구성을 한 번에 처리
            content_parts = [
                # 제목 (가중치 증가)
                *([f"titles: {title_text}", title_text] if (title_text := ' / '.join([
                    title for title in [manga['title'], manga['title_english'], manga['title_japanese']] 
                    if title.strip()
                ])) else []),

                # 리스트 필드들
                *[f"{name}: {' / '.join(value)}" 
                  for name, value in [
                      ('genres', manga['genres']),
                      ('themes', manga['themes']),
                      ('authors', manga['authors']),
                      ('demographics', manga['demographics'])
                  ] if value],
                
                # 텍스트 필드들
                *[f"{name}: {text}"
                  for name, text in [
                      ('synopsis', manga['synopsis'].strip()),
                      ('background', manga['background'].strip())
                  ] if text]
            ]
            
            content = "\n".join(content_parts)
            
            doc = Document(
                page_content=content,
                metadata={
                    'manga_id': manga['id'],
                    'title': manga['title'],
                    'title_english': manga['title_english'],
                    'title_japanese': manga['title_japanese'],
                    'status': manga['status'],
                    'published_start': manga['published_start'],    # 새로운 필드
                    'published_end': manga['published_end'],        # 새로운 필드
                    'background': manga['background'],
                    'genres': manga['genres'],
                    'themes': manga['themes'],
                    'demographics': manga['demographics'],
                    'authors': manga['authors'],
                    'image_url': manga['image_url'],
                    'created_at': manga['created_at']
                }
            )
            documents.append(doc)
            
            if i == 0:
                logger.info(f"첫 번째 문서 내용 길이: {len(content)} chars")
        
        logger.info(f"총 {len(documents)}개 문서 생성 완료")
        return documents


class CSVMangaDataSource(MangaDataSource):
    """CSV 파일에서 만화 데이터를 로드하는 구현체 (정규화 적용)"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._total_count = 0
        self._df_processed = None

    def _load_and_preprocess(self):
        """CSV 로드 및 전처리"""
        if self._df_processed is not None:
            return
        
        try:
            logger.info(f"CSV 로드 및 전처리 시작: {self.csv_path}")
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"CSV 로드 완료: {len(df)} rows")
            
            # CSV 데이터 구조 확인
            logger.info(f"CSV 컬럼들: {list(df.columns)}")
            
            self._total_count = len(df)
            self._df_processed = df
            
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            raise
    
    def get_total_count(self) -> int:
        self._load_and_preprocess()
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """CSV 데이터를 정규화하여 배치 스트리밍"""
        self._load_and_preprocess()
        
        total_rows = len(self._df_processed)
        logger.info(f"CSV 배치 스트리밍 시작: {total_rows} rows, batch size: {batch_size}")
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = self._df_processed.iloc[start_idx:end_idx]
            
            # ✅ 원시 데이터를 정규화하여 반환
            normalized_batch = []
            for raw_record in batch_df.to_dict('records'):
                normalized_record = self._normalize_manga_record(raw_record)
                normalized_batch.append(normalized_record)
            
            logger.info(f"CSV 배치 {start_idx//batch_size + 1}: {len(normalized_batch)} 정규화된 레코드")
            yield normalized_batch


class DatabaseMangaDataSource(MangaDataSource):
    """데이터베이스에서 만화 데이터를 로드하는 구현체 (정규화 적용)"""
    
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
                # TODO: 실제 DB 쿼리 구현
                # connection = self._get_connection()
                # cursor = connection.cursor()
                # total_query = "SELECT COUNT(*) FROM manga_data WHERE status IN ('Finished', 'Publishing')"
                # self._total_count = cursor.execute(total_query).fetchone()[0]
                
                raise NotImplementedError("Database total count not implemented yet")
                
            except Exception as e:
                logger.error(f"총 개수 조회 실패: {e}")
                raise
        
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """데이터베이스에서 정규화된 데이터를 배치 스트리밍"""
        logger.info(f"Database 배치 스트리밍 시작: batch size {batch_size}")
        
        try:
            # TODO: 실제 DB 쿼리 구현
            # connection = self._get_connection()
            # cursor = connection.cursor()
            # 
            # base_query = """
            #     SELECT id, created_at, images, title, status, published, synopsis, 
            #            background, genres, themes, demographics, authors, 
            #            title_english, title_japanese
            #     FROM manga_data 
            #     WHERE status IN ('Finished', 'Publishing')
            #     ORDER BY id
            # """
            # 
            # cursor.execute(base_query)
            # 
            # batch_count = 0
            # while True:
            #     batch_rows = cursor.fetchmany(batch_size)
            #     if not batch_rows:
            #         break
            #     
            #     # ✅ DB 데이터도 정규화하여 반환
            #     normalized_batch = []
            #     for row in batch_rows:
            #         raw_record = dict(row)
            #         normalized_record = self._normalize_manga_record(raw_record)
            #         normalized_batch.append(normalized_record)
            #     
            #     batch_count += 1
            #     logger.info(f"DB Batch {batch_count}: {len(normalized_batch)} 정규화된 레코드")
            #     yield normalized_batch
            
            raise NotImplementedError("Database batch streaming not implemented yet")
            
        except Exception as e:
            logger.error(f"Database 배치 스트리밍 실패: {e}")
            raise


class MockDatabaseMangaDataSource(MangaDataSource):
    """정규화된 형태의 모킹 대용량 데이터베이스 (개발/테스트용)"""
    
    def __init__(self, record_count: int = 1000000):
        self.record_count = record_count
        self._total_count = record_count
        
        # 정규화된 템플릿 데이터
        self.genres_list = [
            ["Action", "Adventure", "Shounen"],
            ["Romance", "Drama", "Josei"],
            ["Horror", "Psychological", "Seinen"],
            ["Comedy", "School", "Shoujo"],
            ["Fantasy", "Magic", "Adventure"],
            ["Award Winning", "Drama", "Mystery"],
            ["Sci-Fi", "Slice of Life"],
            ["Sports"],
            ["Supernatural"]
        ]
        self.themes_list = [
            ["Friendship", "Coming of Age"],
            ["Love Triangle", "School Life"],
            ["Survival", "Gore"],
            ["Slice of Life", "Music"],
            ["Time Travel", "Supernatural"],
            ["Adult Cast", "Psychological"],
            ["Iyashikei"],
            ["Combat Sports"],
            ["Military", "Psychological"]
        ]
        self.demographics = ["Shounen", "Shoujo", "Seinen", "Josei"]
        self.statuses = ["Finished", "Publishing", "On Hiatus"]
        self.authors_list = [
            ["Takahashi, Rumiko"],
            ["Arakawa, Hiromu"],
            ["Kishimoto, Masashi"],
            ["Toriyama, Akira"],
            ["Oda, Eiichiro"],
            ["Urasawa, Naoki"],
            ["Miura, Kentarou"],
            ["CLAMP"],
            ["Kubo, Tite"]
        ]
    
    def get_total_count(self) -> int:
        return self._total_count
    
    def load_manga_data_batches(self, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """정규화된 Mock 데이터를 배치 스트리밍"""
        logger.info(f"Mock 데이터 스트리밍 시작: {self.record_count:,} records, batch size: {batch_size}")
        
        total_batches = (self.record_count + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_id = batch_num * batch_size + 1
            end_id = min((batch_num + 1) * batch_size, self.record_count)
            batch_data = []
            
            for i in range(start_id, end_id + 1):
                idx = i % len(self.genres_list)
                
                # ✅ 바로 정규화된 형태로 생성
                normalized_record = {
                    'id': i,
                    'title': f'Mock Manga {i:06d}',
                    'title_english': f'Mock Manga {i:06d}',
                    'title_japanese': f'モックマンガ {i:06d}',
                    'status': self.statuses[i % len(self.statuses)],
                    'synopsis': f'This is a mock synopsis for manga {i:06d}. It features an exciting story in the {self.genres_list[idx][0]} genre with {self.themes_list[idx][0]} themes.',
                    'background': f'Mock background information for manga {i:06d}.',
                    'created_at': '2025-07-19 10:00:00.000000+00',
                    
                    # 이미 정규화된 리스트 형태
                    'genres': self.genres_list[idx],
                    'themes': self.themes_list[idx],
                    'demographics': [self.demographics[i % len(self.demographics)]],
                    'authors': self.authors_list[idx],
                    
                    # 정규화된 문자열
                    'image_url': f"https://cdn.example.com/images/manga/{i}.jpg",
                    'published': f"Jan 1, {2000 + (i % 20)} to Dec 31, {2001 + (i % 20)}"
                }
                batch_data.append(normalized_record)
            
            logger.info(f"Mock 배치 {batch_num + 1}/{total_batches}: {len(batch_data)} 정규화된 레코드")
            yield batch_data
        
        logger.info(f"Mock 데이터 스트리밍 완료: {total_batches} 배치, {self.record_count:,} 레코드")