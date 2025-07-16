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
        profile: Dict[str, Any],
        candidates: List[Document],
        max_candidates: int = 15
    ) -> str:
        """추천 생성을 위한 프롬프트 생성"""
        
        prompt = """당신은 만화 추천 전문가입니다.

[사용자가 좋아하는 만화의 특징]
"""
        
        # 좋아하는 만화들의 정보 추가
        for doc in favorite_docs:
            if hasattr(doc, 'metadata'):
                title = doc.metadata.get('title', '제목 없음')
                genre = doc.metadata.get('main_genre_cd_nm', 'N/A')
                author = doc.metadata.get('author', 'N/A')
                outline = doc.metadata.get('outline', '')
                web_info = doc.metadata.get('web_info', '')
                
                prompt += f"""
• {title}
  장르: {genre}
  작가: {author}
  줄거리: {outline[:150]}...
  웹 검색 정보: {web_info[:200] if web_info else '추가 정보 없음'}...
"""
        
        # 사용자 프로필 정보 추가
        prompt += f"""

[사용자 프로필]
- 좋아하는 만화: {', '.join(profile['favorite_manga'])}
- 선호 장르: {', '.join(profile['preferred_genres'])}
- 연령: {profile['age_group'][0]} ~ {profile['age_group'][1]}세
- 성별: {profile['gender']}

[추천 후보 만화 목록]
"""
        
        # 후보 만화들에 번호 매기기
        limited_candidates = candidates[:max_candidates]
        for i, doc in enumerate(limited_candidates, 1):
            m = doc.metadata
            prompt += f"""
{i}. {m['title']}
   장르: {m.get('main_genre_cd_nm', 'N/A')}
   작가: {m.get('author', 'N/A')}
   줄거리: {m.get('outline', 'N/A')[:200]}...
   웹 검색 정보: {m.get('web_info', '추가 정보 없음')[:200]}...
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
        profile: Dict[str, Any],
        recommendations: List[Dict],
        candidates: List[Document]
    ) -> str:
        """추천 결과 검증을 위한 프롬프트 생성"""
        
        prompt = f"""당신은 만화 추천 시스템의 품질 검증 전문가입니다.

[사용자 프로필]
- 좋아하는 만화: {', '.join(profile['favorite_manga'])}
- 연령: {profile['age_group'][0]} ~ {profile['age_group'][1]}세
- 선호 장르: {', '.join(profile['preferred_genres'])}

[추천 결과 (총 {len(recommendations)}개)]
"""
        
        # 추천 결과들 나열
        for i, rec in enumerate(recommendations, 1):
            # 인덱스에서 실제 정보 가져오기
            doc = candidates[rec['index'] - 1]
            title = doc.metadata.get('title', 'N/A')
            reason = rec['recommendation_reason']
            prompt += f"{i}. {title} - {reason}\n"
        
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
        """웹 검색을 위한 쿼리 생성"""
        return f"{manga_title} 만화" 