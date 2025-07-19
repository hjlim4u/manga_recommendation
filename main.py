import asyncio
from typing import Dict, Optional
from data_source import CSVMangaDataSource, MockDatabaseMangaDataSource, DatabaseMangaDataSource
from manga_recommendation_langgraph import create_recommendation_graph  # 🎯 간단한 import!

async def main():
    """메인 추천 실행 함수"""
    print("Hello from manga-recommendation!")
    genres = ["Action","Adventure","Avant Garde","Award Winning","Boys Love","Comedy","Drama","Ecchi","Erotica","Fantasy","Girls Love","Gourmet","Hentai","Horror","Mystery","Romance","Sci-Fi","Slice of Life","Sports","Supernatural","Suspense"]
    demographics = ["Josei","Kids","Seinen","Shoujo","Shounen"]
    # 사용 예시
    user_input = {
        "gender": "female",
        "age": "18~30", 
        "genres": ["Avant Garde", "Drama"],
        "favorite_manga": "Kaibutsu Oujo"
    }
    
    # 데이터 소스 선택 (예시)
    # 방법 1: CSV 파일 사용 (테스트/개발용)
    csv_data_source = CSVMangaDataSource("manga_rows.csv")
    
    # 방법 2: 모킹 데이터베이스 사용 (대용량 테스트용) 
    # mock_data_source = MockDatabaseMangaDataSource(record_count=100000)
    
    # 방법 3: 실제 데이터베이스 사용 (상용)
    # db_config = {"host": "localhost", "database": "manga_db", "user": "user", "password": "pass"}
    # db_data_source = DatabaseMangaDataSource(db_config)
    
    # 추천 그래프 생성 및 실행
    app = create_recommendation_graph(csv_data_source)
    
    initial_state = {
        "user_gender": user_input['gender'],
        "user_age_group": user_input['age'],
        "user_genres": user_input['genres'],
        "user_favorite_manga": user_input['favorite_manga'],
        "favorite_manga_docs": [],
        "search_results": [],
        "search_attempt": 0,
        "recommendations": [],
        "recommendation_quality": 0.0,
        "needs_refinement": False,
        "validation_log": []
    }
    
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 10})
    
    # 결과 출력
    print("\n=== 추천 결과 ===")
    recommendations = final_state['recommendations']
    candidates = final_state['search_results']
    
    # 인덱스 방식 추천을 완전한 정보로 변환하여 출력
    for i, rec in enumerate(recommendations, 1):
        doc = candidates[rec['index'] - 1]
        
        print(f"\n{i}. {doc.metadata['title']} ({doc.metadata.get('genres', 'N/A')})")
        print(f"   작가: {doc.metadata.get('author', 'N/A')}")
        print(f"   유사도: {doc.metadata.get('similarity_score', 0):.3f}")
        print(f"   추천 이유: {rec['recommendation_reason']}")
    
    print("\n=== 처리 로그 ===")
    for log in final_state['validation_log']:
        print(f"- {log}")
    
    print(f"\n최종 품질 점수: {final_state['recommendation_quality']*100:.0f}/100")

if __name__ == "__main__":
    asyncio.run(main())
