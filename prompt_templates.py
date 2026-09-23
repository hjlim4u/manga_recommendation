"""
만화 추천 시스템의 프롬프트 템플릿 관리 모듈
"""
from typing import Dict, List
from langchain_core.documents import Document


def _fmt(doc: Document) -> str:
    """만화 정보 + 웹 검색 정보를 프롬프트용 텍스트로 포맷"""
    web_info = doc.metadata.get('web_info', '')
    return f"""만화 정보:
{doc.page_content}

웹 검색 추가 정보: {web_info[:200] if web_info else '추가 정보 없음'}..."""


def generate_recommendation_prompt(
    favorite_docs: List[Document],
    gender: str,
    age_group: str,
    preferred_genres: List[str],
    candidates: List[Document],
    max_candidates: int = 15
) -> str:
    """추천 생성을 위한 프롬프트 생성"""
    favorites = "\n\n".join(f"• {_fmt(doc)}" for doc in favorite_docs)
    limited_candidates = candidates[:max_candidates]
    candidates_text = "\n\n".join(f"{i}. {_fmt(doc)}" for i, doc in enumerate(limited_candidates, 1))

    return f"""당신은 만화 추천 전문가입니다.

[사용자가 좋아하는 만화의 특징]
{favorites}

[사용자 프로필]
- 선호 장르: {', '.join(preferred_genres)}
- 연령대: {age_group}
- 성별: {gender}

[추천 후보 만화 목록]
{candidates_text}

위 {len(limited_candidates)}개 후보 중 사용자에게 가장 적합한 3개를 선택하여 추천해주세요.
각 추천마다 사용자가 좋아하는 만화와의 구체적인 연관성과 추천 이유를 설명하세요.

**중요**: 반드시 정확히 3개의 추천을 제공해야 하며, 아래 JSON 형식으로만 응답하세요.
번호는 위 목록의 번호를 정확히 사용하세요.

{{
  "recommendations": [
    {{"index": 번호, "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"}},
    {{"index": 번호, "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"}},
    {{"index": 번호, "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"}}
  ]
}}

JSON 형식 외의 다른 텍스트는 포함하지 마세요."""


def generate_validation_prompt(
    favorite_docs: List[Document],
    gender: str,
    age_group: str,
    preferred_genres: List[str],
    recommendations: List[Dict],
    candidates: List[Document]
) -> str:
    """추천 결과 검증을 위한 프롬프트 생성"""
    favorites = "\n\n".join(f"• {_fmt(doc)}" for doc in favorite_docs)

    recs_text = "\n\n".join(
        f"{i}. 추천 만화:\n{candidates[rec['index'] - 1].page_content}\n\n추천 이유: {rec['recommendation_reason']}"
        for i, rec in enumerate(recommendations, 1)
    )

    return f"""당신은 만화 추천 시스템의 품질 검증 전문가입니다.

[사용자 프로필]
- 연령대: {age_group}
- 선호 장르: {', '.join(preferred_genres)}
- 사용자가 좋아하는 만화 정보:
{favorites}

[추천 결과 (총 {len(recommendations)}개)]
{recs_text}

3개의 추천이 모두 제공되었고, 사용자의 취향에 적합한지 평가하세요.
추천 품질을 평가하고 JSON 형식으로 응답하세요:

{{
    "score": 0-100 사이의 점수,
    "pass": true/false (75점 이상이면 true),
    "reasoning": "3개 추천 모두에 대한 평가 근거를 2-3문장으로"
}}
"""


def generate_web_search_query(manga_title: str) -> str:
    """웹 검색을 위한 쿼리 생성"""
    return f"{manga_title} manga"
