"""CSV에서 만화 데이터를 로드하고 정규화하는 모듈"""
import csv
import json
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _normalize_json_field(value: Any) -> List[str]:
    """JSON 필드를 일관된 리스트 형식으로 정규화"""
    if not value:
        return []
    try:
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, list):
            return value
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _extract_image_url(images_data: Any) -> str:
    """이미지 데이터에서 URL 추출 (JPG 우선, 없으면 WebP)"""
    if not images_data:
        return ""
    try:
        images = json.loads(images_data) if isinstance(images_data, str) else images_data
        if isinstance(images, dict):
            return images.get('jpg', {}).get('image_url') or images.get('webp', {}).get('image_url') or ""
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _extract_published_info(published_data: Any) -> tuple[str, str]:
    """출간 정보에서 from, to 날짜 추출 (예: "1994-12-05T00:00:00+00:00" -> "1994-12-05")"""
    if not published_data:
        return "", ""
    try:
        pub = json.loads(published_data) if isinstance(published_data, str) else published_data
        if not isinstance(pub, dict):
            return "", ""
        return (pub.get('from') or '')[:10], (pub.get('to') or '')[:10]
    except (json.JSONDecodeError, TypeError):
        return "", ""


def _normalize_manga_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """원시 CSV 레코드를 정규화된 형식으로 변환"""
    published_start, published_end = _extract_published_info(raw_record.get('published'))

    return {
        'id': int(raw_record.get('id', 0)),
        'title': str(raw_record.get('title', '')),
        'title_english': str(raw_record.get('title_english', '')),
        'title_japanese': str(raw_record.get('title_japanese', '')),
        'status': str(raw_record.get('status', '')),
        'synopsis': str(raw_record.get('synopsis', '')),
        'background': str(raw_record.get('background', '')),
        'created_at': str(raw_record.get('created_at', '')),
        'genres': _normalize_json_field(raw_record.get('genres')),
        'themes': _normalize_json_field(raw_record.get('themes')),
        'demographics': _normalize_json_field(raw_record.get('demographics')),
        'authors': _normalize_json_field(raw_record.get('authors')),
        'image_url': _extract_image_url(raw_record.get('images')),
        'published_start': published_start,
        'published_end': published_end,
    }


def load_manga_data(csv_path: str) -> List[Dict[str, Any]]:
    """CSV 파일을 로드하고 정규화된 레코드 리스트로 반환"""
    with open(csv_path, encoding='utf-8', newline='') as f:
        records = [_normalize_manga_record(row) for row in csv.DictReader(f)]
    logger.info(f"CSV 로드 완료: {len(records)} rows ({csv_path})")
    return records


def create_documents(manga_data: List[Dict[str, Any]]) -> List[Document]:
    """정규화된 데이터로부터 Document 생성"""
    documents = []

    for manga in manga_data:
        title_text = ' / '.join(
            title for title in [manga['title'], manga['title_english'], manga['title_japanese']]
            if title.strip()
        )

        content_parts = [
            *([f"titles: {title_text}", title_text] if title_text else []),
            *[f"{name}: {' / '.join(value)}"
              for name, value in [
                  ('genres', manga['genres']),
                  ('themes', manga['themes']),
                  ('authors', manga['authors']),
                  ('demographics', manga['demographics'])
              ] if value],
            *[f"{name}: {text}"
              for name, text in [
                  ('synopsis', manga['synopsis'].strip()),
                  ('background', manga['background'].strip())
              ] if text]
        ]

        documents.append(Document(
            page_content="\n".join(content_parts),
            metadata={'manga_id': manga['id'], **{k: v for k, v in manga.items() if k not in ('id', 'synopsis')}}
        ))

    logger.info(f"총 {len(documents)}개 문서 생성 완료")
    return documents
