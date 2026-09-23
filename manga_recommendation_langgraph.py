# ===== 임포트 =====
from typing import Any, Dict, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
import asyncio
import json
import re
from dotenv import load_dotenv
from vector_store import QdrantMangaStore
import prompt_templates

# 환경 변수 로드
load_dotenv()

# ===== State 정의 =====
class RecommendationState(TypedDict):
    """추천 시스템의 상태"""
    # 사용자 입력
    user_gender: Literal["male", "female", "skip"]
    user_age_group: Literal["12~15", "15~18", "18~30", "30~40", "40~50", "50~"]
    user_genres: List[str]
    user_favorite_manga: str  # 단수 형태

    # 좋아하는 만화 Document (효율적 저장)
    favorite_manga_docs: List[Document]

    # 검색 결과
    search_results: List[Document]

    # 추천 결과
    recommendations: List[Dict]
    recommendation_quality: float
    attempt: int  # 추천 생성 재시도 횟수

    # 검증 관련
    needs_refinement: bool


# ===== 메인 노드 클래스 =====
class MangaRecommendationNodes:
    def __init__(self, manga_data: List[Dict[str, Any]]):
        # 벡터 스토어
        self.vector_store = QdrantMangaStore()

        # LLM
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")

        # 웹 검색
        self.web_search_tool = TavilySearch(max_results=3)

        # 데이터 초기화
        self.vector_store.load_and_index_from_source(manga_data)

    def process_user_profile(self, state: RecommendationState) -> RecommendationState:
        # 벡터 DB를 활용한 좋아하는 만화 검색
        found_doc = self.vector_store.find_manga_by_title(state['user_favorite_manga'])
        state['favorite_manga_docs'] = [found_doc] if found_doc else []
        state['attempt'] = 0

        return state

    def search_similar_manga(self, state: RecommendationState) -> RecommendationState:
        """벡터 검색"""
        state['search_results'] = self.vector_store.search_similar_manga(
            favorite_manga_docs=state['favorite_manga_docs'],
            limit=30
        )
        return state

    async def enrich_with_web_search(self, state: RecommendationState) -> RecommendationState:
        """웹 검색으로 데이터 보강"""
        candidates = state['search_results'][:8]

        async def search_manga_info(doc):
            try:
                query = prompt_templates.generate_web_search_query(doc.metadata['title'])
                result = await asyncio.to_thread(
                    self.web_search_tool.invoke,
                    {"query": query}
                )

                if result and 'results' in result:
                    summaries = [r['content'][:200] for r in result['results'][:2] if 'content' in r]
                    doc.metadata['web_info'] = ' '.join(summaries)
                else:
                    doc.metadata['web_info'] = ''
            except Exception:
                doc.metadata['web_info'] = ''

            return doc

        # 선호 만화 검색 (최대 2개)
        favorite_tasks = [search_manga_info(doc) for doc in state['favorite_manga_docs'][:2]]
        candidate_tasks = [search_manga_info(doc) for doc in candidates]

        state['favorite_manga_docs'] = await asyncio.gather(*favorite_tasks)
        state['search_results'] = await asyncio.gather(*candidate_tasks)

        return state

    def generate_recommendations(self, state: RecommendationState) -> RecommendationState:
        """LLM 기반 추천 생성 - 인덱스 기반"""
        candidates = state['search_results']
        state['attempt'] = state.get('attempt', 0) + 1

        if not candidates:
            state['recommendations'] = []
            return state

        prompt = prompt_templates.generate_recommendation_prompt(
            favorite_docs=state.get('favorite_manga_docs', []),
            preferred_genres=state['user_genres'],
            age_group=state['user_age_group'],
            gender=state['user_gender'],
            candidates=candidates
        )

        response = self.llm.invoke(prompt)

        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if not json_match:
                raise ValueError("JSON을 찾을 수 없음")

            llm_recommendations = json.loads(json_match.group()).get('recommendations', [])
            recommendations = self._create_recommendations_from_indices(llm_recommendations, candidates)
        except Exception:
            # 빈 배열로 초기화 - validate_results에서 재시도 처리
            recommendations = []

        state['recommendations'] = recommendations

        return state

    def _create_recommendations_from_indices(self, llm_recommendations: List[Dict], candidates: List[Document]) -> List[Dict]:
        """인덱스 기반으로 추천 생성 (인덱스 + 이유만 저장)"""
        recommendations = []
        used_indices = set()

        for llm_rec in llm_recommendations:
            index = llm_rec.get('index', 0)
            if 1 <= index <= len(candidates) and index not in used_indices:
                used_indices.add(index)
                recommendations.append({
                    'index': index,
                    'recommendation_reason': llm_rec.get('reason', '추천')
                })

        # 3개 미만인 경우 상위 후보로 채우기
        fill = [
            {'index': i, 'recommendation_reason': "높은 유사도를 바탕으로 한 추천"}
            for i in range(1, len(candidates) + 1) if i not in used_indices
        ]
        recommendations += fill[:3 - len(recommendations)]

        return recommendations[:3]

    def validate_results(self, state: RecommendationState) -> RecommendationState:
        """JSON 기반 추천 결과 검증 - 3개 추천 버전 (인덱스 방식)"""
        recommendations = state['recommendations']
        candidates = state['search_results']

        # 핵심: 최대 시도 횟수를 먼저 체크하여 무한 루프 방지
        if state.get('attempt', 0) >= 2:
            state['needs_refinement'] = False
            state['recommendation_quality'] = 0.8  # 기본 점수
            return state

        if len(recommendations) < 3:
            state['recommendation_quality'] = 0.6  # 낮은 점수
            state['needs_refinement'] = True
            return state

        validation_prompt = prompt_templates.generate_validation_prompt(
            age_group=state['user_age_group'],
            gender=state['user_gender'],
            favorite_docs=state.get('favorite_manga_docs', []),
            preferred_genres=state['user_genres'],
            recommendations=recommendations,
            candidates=candidates
        )

        try:
            response = self.llm.invoke(validation_prompt)
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                state['recommendation_quality'] = result.get('score', 0) / 100.0
                state['needs_refinement'] = not result.get('pass', False)
            else:
                # 파싱 실패 시 기본 통과
                state['recommendation_quality'] = 0.8
                state['needs_refinement'] = False
        except Exception:
            state['recommendation_quality'] = 0.8
            state['needs_refinement'] = False

        return state


# ===== 그래프 구성 =====
def create_recommendation_graph(manga_data: List[Dict[str, Any]]) -> StateGraph:
    """추천 시스템 그래프 생성 - sequential 메소드 활용"""

    nodes = MangaRecommendationNodes(manga_data)

    workflow = StateGraph(RecommendationState)

    # 데이터 처리 및 검색 파이프라인
    workflow.add_sequence([
        nodes.process_user_profile,     # 사용자 프로필 처리
        nodes.search_similar_manga,     # 벡터 검색
        nodes.enrich_with_web_search    # 웹 검색으로 데이터 보강
    ])

    # 추천 생성 및 검증 파이프라인
    workflow.add_sequence([
        nodes.generate_recommendations, # 추천 생성
        nodes.validate_results          # 결과 검증
    ])

    # 엣지 정의 - 시퀀스 간 연결
    workflow.set_entry_point("process_user_profile")
    workflow.add_edge("enrich_with_web_search", "generate_recommendations")

    # 조건부 엣지 - 재시도 로직 (검색은 이미 결정적이므로 재생성만 재시도)
    def should_retry_or_end(state: RecommendationState) -> str:
        if state.get('needs_refinement', False):
            return "retry_generation"
        return "end"

    workflow.add_conditional_edges(
        "validate_results",
        should_retry_or_end,
        {
            "retry_generation": "generate_recommendations",
            "end": END
        }
    )

    return workflow.compile()

# ===== 메인 실행 (테스트용) =====
if __name__ == "__main__":
    print("이 파일은 라이브러리 모듈입니다. main.py를 실행해주세요.")
