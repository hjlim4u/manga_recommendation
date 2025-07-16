from typing import Dict, List, TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
import asyncio
import json
import re
from dotenv import load_dotenv
import nest_asyncio
from domain import Gender, AgeGroup, AgeRating
from vector_store import QdrantMangaStore
from data_source import MangaDataSource, CSVMangaDataSource, MockDatabaseMangaDataSource, DatabaseMangaDataSource
from prompt_templates import PromptTemplates

nest_asyncio.apply()
# 환경 변수 로드
load_dotenv()

# ===== State 정의 =====
class RecommendationState(TypedDict):
    """추천 시스템의 상태"""
    # 사용자 입력
    user_gender: Literal["남", "여", "넘어가기"]
    user_age_group: Literal["12~15", "15~18", "18~30", "30~40", "40~50", "50~"]
    user_genres: List[str]
    user_favorite_manga: str
    
    # 처리된 프로필
    processed_profile: Dict
    
    # 🔧 좋아하는 만화 Document들 (효율적 저장)
    favorite_manga_docs: List[Document]
    
    # 검색 결과
    search_results: List[Document]
    search_attempt: int

    # 추천 결과
    recommendations: List[Dict]
    recommendation_quality: float
    
    # 검증 관련
    needs_refinement: bool



# ===== 메인 노드 클래스 =====
class MangaRecommendationNodes:
    def __init__(self, data_source: MangaDataSource):
        # 데이터 소스 저장
        self.data_source = data_source
        
        # 벡터 스토어
        self.vector_store = QdrantMangaStore()
        
        # LLM
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        
        # 웹 검색
        self.web_search_tool = TavilySearch(max_results=3)
        
        # 데이터 초기화
        self._initialize_data()
    
    def _initialize_data(self):
        """데이터 소스에서 데이터 로드 및 벡터 DB 인덱싱"""
        try:
            # 데이터 소스를 통해 로드 및 인덱싱
            self.vector_store.load_and_index_from_source(self.data_source)
            
        except Exception as e:
            raise

    
    def process_user_profile(self, state: RecommendationState) -> RecommendationState:
        """사용자 입력을 프로필로 변환 + 좋아하는 만화 검색"""
        gender = Gender.from_option(state['user_gender'])
        age_group = AgeGroup.from_option(state['user_age_group'])
        
        profile = {
            'gender': gender.value,
            'age_group': age_group.value,
            'preferred_genres': state['user_genres'],
            'favorite_manga': self._parse_favorite_manga(state['user_favorite_manga']),
            'max_age_rating': age_group.min_age
        }
        
        # 🔧 벡터 DB를 활용한 좋아하는 만화 검색 (메모리 효율적)
        favorite_manga_docs = []
        not_found_titles = []
        
        for title in profile['favorite_manga']:
            found_doc = self.vector_store.find_manga_by_title(title)
            if found_doc:
                favorite_manga_docs.append(found_doc)
            else:
                not_found_titles.append(title)
        
        state['processed_profile'] = profile
        state['favorite_manga_docs'] = favorite_manga_docs  # 🔧 상태에 저장
        state['search_attempt'] = 0
        
        return state
    
    
    def _parse_favorite_manga(self, manga_text: str) -> List[str]:
        """좋아하는 만화 텍스트 파싱"""
        import re
        manga_list = re.split(r'[,/\n]|과\s|와\s|랑\s', manga_text)
        return [m.strip() for m in manga_list if m.strip() and len(m.strip()) > 1]
    
    async def search_similar_manga(self, state: RecommendationState) -> RecommendationState:
        """벡터 검색 (최대 2회 시도)"""
        profile = state['processed_profile']
        favorite_manga_docs = state['favorite_manga_docs']  # 🔧 상태에서 가져오기
        attempt = state.get('search_attempt', 0)
        
        if attempt == 0:
            # 전략 1: 중심점 임베딩
            results = self.vector_store.search_similar_manga_by_centroid(
                favorite_manga_docs=favorite_manga_docs,
                preferred_genres=profile['preferred_genres'],
                max_age_rating=profile['max_age_rating'],
                limit=30
            )
        else:
            # 전략 2: 개별 검색 후 병합
            results = self.vector_store.search_similar_manga_by_individual(
                favorite_manga_docs=favorite_manga_docs,
                preferred_genres=profile['preferred_genres'],
                max_age_rating=profile['max_age_rating'],
                limit_per_manga=15
            )
        
        state['search_results'] = results
        state['search_attempt'] = attempt + 1
        
        return state
    
    async def enrich_with_web_search(self, state: RecommendationState) -> RecommendationState:
        """웹 검색으로 데이터 보강"""
        candidates = state['search_results'][:8]
        profile = state['processed_profile']
        
        # 1. 선호 만화 정보 수집
        favorite_manga_info = {}
        
        # 2. 후보 만화 검색
        async def search_manga_info(doc):
            try:
                query = PromptTemplates.generate_web_search_query(doc.metadata['title'])
                result = await asyncio.to_thread(
                    self.web_search_tool.invoke,
                    {"query": query}
                )
                
                if result and 'results' in result:
                    summaries = []
                    for r in result['results'][:2]:
                        if 'content' in r:
                            summaries.append(r['content'][:200])
                    doc.metadata['web_info'] = ' '.join(summaries)
                else:
                    doc.metadata['web_info'] = ''
            except Exception as e:
                doc.metadata['web_info'] = ''
            
            return doc
        # 선호 만화 검색 (최대 2개)
        favorite_tasks = [search_manga_info(doc) for doc in state['favorite_manga_docs'][:2]]
        favorite_manga_info = await asyncio.gather(*favorite_tasks)
        
        candidate_tasks = [search_manga_info(doc) for doc in candidates]
        enriched_candidates = await asyncio.gather(*candidate_tasks)
        
        state['favorite_manga_docs'] = favorite_manga_info
        state['search_results'] = enriched_candidates
        
        return state
    
    def generate_recommendations(self, state: RecommendationState) -> RecommendationState:
        """LLM 기반 추천 생성 - 인덱스 기반"""
        candidates = state['search_results']
        profile = state['processed_profile']
        favorite_docs = state.get('favorite_manga_docs', [])
        
        if not candidates:
            state['recommendations'] = []
            return state
        
        # 프롬프트 템플릿을 사용하여 생성
        prompt = PromptTemplates.generate_recommendation_prompt(
            favorite_docs=favorite_docs,
            profile=profile,
            candidates=candidates
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            # JSON 파싱
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                llm_recommendations = result.get('recommendations', [])
            else:
                raise ValueError("JSON을 찾을 수 없음")
            
            # 인덱스 기반으로 추천 생성
            recommendations = self._create_recommendations_from_indices(
                llm_recommendations, candidates
            )
            
        except Exception as e:
            # 빈 배열로 초기화 - validate_results에서 재시도 처리
            recommendations = []
        
        # 🔍 최종 추천에 좋아하는 만화가 포함되었는지 확인
        favorite_manga_titles = [doc.metadata.get('title') for doc in state.get('favorite_manga_docs', []) if hasattr(doc, 'metadata')]
        favorite_manga_ids = [doc.metadata.get('manga_id') for doc in state.get('favorite_manga_docs', []) if hasattr(doc, 'metadata')]
        
        state['recommendations'] = recommendations
        state['search_results'] = candidates  # candidates도 state에 저장 (나중에 사용하기 위해)
        
        return state
    
    def _create_recommendations_from_indices(self, llm_recommendations: List[Dict], candidates: List[Document]) -> List[Dict]:
        """인덱스 기반으로 추천 생성 - 매우 간단! (인덱스 + 이유만 저장)"""
        recommendations = []
        used_indices = set()
        
        for llm_rec in llm_recommendations:
            index = llm_rec.get('index', 0)
            reason = llm_rec.get('reason', '추천')
            
            # 인덱스 유효성 검사
            if 1 <= index <= len(candidates) and index not in used_indices:
                used_indices.add(index)
                
                # 인덱스와 이유만 저장 (메타데이터 중복 제거)
                rec = {
                    'index': index,
                    'recommendation_reason': reason
                }
                recommendations.append(rec)

        
        # 3개 미만인 경우 상위 후보로 채우기
        while len(recommendations) < 3:
            for i in range(1, len(candidates) + 1):
                if i not in used_indices:
                    rec = {
                        'index': i,
                        'recommendation_reason': "높은 유사도를 바탕으로 한 추천"
                    }
                    recommendations.append(rec)
                    used_indices.add(i)
                    
                    break
            else:
                break
        
        return recommendations[:3]
    
    def validate_results(self, state: RecommendationState) -> RecommendationState:
        """JSON 기반 추천 결과 검증 - 3개 추천 버전 (인덱스 방식)"""
        recommendations = state['recommendations']
        candidates = state['search_results']
        profile = state['processed_profile']
        
        # 🔑 핵심: 최대 시도 횟수를 먼저 체크하여 무한 루프 방지
        if state.get('search_attempt', 0) >= 2:
            state['needs_refinement'] = False
            state['recommendation_quality'] = 0.8  # 기본 점수
            return state
        
        if len(recommendations) < 3:
            state['recommendation_quality'] = 0.6  # 낮은 점수
            state['needs_refinement'] = True  
            return state
        
        # 프롬프트 템플릿을 사용하여 생성
        validation_prompt = PromptTemplates.generate_validation_prompt(
            profile=profile,
            recommendations=recommendations,
            candidates=candidates
        )
        
        try:
            response = self.llm.invoke(validation_prompt)
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                state['recommendation_quality'] = result.get('score', 0) / 100.0
                state['needs_refinement'] = not result.get('pass', False)
            else:
                # 파싱 실패 시 기본 통과
                state['recommendation_quality'] = 0.8
                state['needs_refinement'] = False
                
        except Exception as e:
            state['recommendation_quality'] = 0.8
            state['needs_refinement'] = False
        
        return state

# ===== 그래프 구성 =====
def create_recommendation_graph(data_source: MangaDataSource) -> StateGraph:
    """추천 시스템 그래프 생성 - sequential 메소드 활용"""
    
    nodes = MangaRecommendationNodes(data_source)
    
    workflow = StateGraph(RecommendationState)
    
    # sequential 메소드로 연속 노드들을 한번에 연결
    # 데이터 처리 및 검색 파이프라인
    workflow.add_sequence([
        nodes.process_user_profile,     # 사용자 프로필 처리
        nodes.search_similar_manga,     # 벡터 검색
        nodes.enrich_with_web_search    # 웹 검색으로 데이터 보강
    ])
    
    # 추천 생성 및 검증 파이프라인
    workflow.add_sequence([
        nodes.generate_recommendations, # 추천 생성
        nodes.validate_results         # 결과 검증
    ])
    
    # 엣지 정의 - 시퀀스 간 연결
    workflow.set_entry_point("process_user_profile")
    workflow.add_edge("enrich_with_web_search", "generate_recommendations")
    
    # 조건부 엣지 - 재시도 로직
    def should_retry_or_end(state: RecommendationState) -> str:
        if state.get('needs_refinement', False):
            return "retry_search"
        return "end"
    
    workflow.add_conditional_edges(
        "validate_results",
        should_retry_or_end,
        {
            "retry_search": "search_similar_manga",  # 재시도 시 벡터 검색부터
            "end": END
        }
    )
    
    return workflow.compile()

# ===== 메인 실행 (테스트용) =====
if __name__ == "__main__":
    """
    테스트 목적으로만 사용. 실제 실행은 main.py를 사용하세요.
    """
    print("이 파일은 라이브러리 모듈입니다. main.py를 실행해주세요.") 