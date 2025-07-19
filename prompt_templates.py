"""
만화 추천 시스템의 프롬프트 템플릿 관리 모듈
"""
from typing import Dict, List, Any
from langchain_core.documents import Document


class PromptTemplates:
    """만화 추천 시스템에서 사용되는 프롬프트 템플릿들을 관리하는 클래스"""
    
    @staticmethod
    def generate_recommendation_prompt(
        favorite_docs: List[Document],
        gender: str,
        age_group: str,
        preferred_genres: List[str],
        candidates: List[Document],
        max_candidates: int = 15
    ) -> str:
        """추천 생성을 위한 프롬프트 생성"""
        
        prompt = """당신은 만화 추천 전문가입니다.

[사용자가 좋아하는 만화의 특징]
"""
        
        # 좋아하는 만화들의 정보 추가 (page_content 사용)
        for doc in favorite_docs:
            web_info = doc.metadata.get('web_info', '')
            
            prompt += f"""
• 만화 정보:
{doc.page_content}

웹 검색 추가 정보: {web_info[:200] if web_info else '추가 정보 없음'}...
"""
        
        # 사용자 프로필 정보 추가
        prompt += f"""

[사용자 프로필]
- 선호 장르: {', '.join(preferred_genres)}
- 연령대: {age_group}
- 성별: {gender}

[추천 후보 만화 목록]
"""
        
        # 후보 만화들에 번호 매기기 (page_content 사용)
        limited_candidates = candidates[:max_candidates]
        for i, doc in enumerate(limited_candidates, 1):
            web_info = doc.metadata.get('web_info', '')
            
            prompt += f"""
{i}. 만화 정보:
{doc.page_content}

웹 검색 추가 정보: {web_info[:200] if web_info else '추가 정보 없음'}...

"""
        
        # 응답 형식 지정
        prompt += f"""

위 {len(limited_candidates)}개 후보 중 사용자에게 가장 적합한 3개를 선택하여 추천해주세요.
각 추천마다 사용자가 좋아하는 만화와의 구체적인 연관성과 추천 이유를 설명하세요.

**중요**: 반드시 정확히 3개의 추천을 제공해야 하며, 아래 JSON 형식으로만 응답하세요.
번호는 위 목록의 번호를 정확히 사용하세요.

{{
  "recommendations": [
    {{
      "index": 번호,
      "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"
    }},
    {{
      "index": 번호,
      "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"
    }},
    {{
      "index": 번호, 
      "reason": "구체적인 추천 이유와 좋아하는 만화와의 연관성"
    }}
  ]
}}

JSON 형식 외의 다른 텍스트는 포함하지 마세요."""
        
        return prompt
    
    @staticmethod
    def generate_validation_prompt(
        favorite_docs: List[Document],
        gender: str,
        age_group: str,
        preferred_genres: List[str],
        recommendations: List[Dict],
        candidates: List[Document]
    ) -> str:
        """추천 결과 검증을 위한 프롬프트 생성"""
        
        prompt = f"""당신은 만화 추천 시스템의 품질 검증 전문가입니다.

[사용자 프로필]
- 연령대: {age_group}
- 선호 장르: {', '.join(preferred_genres)}
- 사용자가 좋아하는 만화 정보:
"""

        # 좋아하는 만화들의 정보 추가 (page_content 사용)
        for doc in favorite_docs:
            web_info = doc.metadata.get('web_info', '')

            prompt += f"""
• 만화 정보:
{doc.page_content}

[추천 결과 (총 {len(recommendations)}개)]
"""
        
        # 추천 결과들 나열 (page_content 사용)
        for i, rec in enumerate(recommendations, 1):
            # 인덱스에서 실제 정보 가져오기
            doc = candidates[rec['index'] - 1]
            reason = rec['recommendation_reason']
            
            prompt += f"""
{i}. 추천 만화:
{doc.page_content}

추천 이유: {reason}

"""
        
        prompt += """
3개의 추천이 모두 제공되었고, 사용자의 취향에 적합한지 평가하세요.
추천 품질을 평가하고 JSON 형식으로 응답하세요:

{
    "score": 0-100 사이의 점수,
    "pass": true/false (75점 이상이면 true),
    "reasoning": "3개 추천 모두에 대한 평가 근거를 2-3문장으로"
}
"""
        
        return prompt
    
    @staticmethod
    def generate_web_search_query(manga_title: str) -> str:
        """웹 검색을 위한 쿼리 생성 (metadata에서 제목 추출)"""
        return f"{manga_title} manga"