import asyncio
from data_source import load_manga_data
from manga_recommendation_langgraph import create_recommendation_graph

async def main():
    """메인 추천 실행 함수"""
    print("Hello from manga-recommendation!")

    user_input = {
        "gender": "female",
        "age": "18~30",
        "genres": ["Avant Garde", "Drama"],
        "favorite_manga": "Kaibutsu Oujo"
    }

    manga_data = load_manga_data("manga_rows.csv")
    app = create_recommendation_graph(manga_data)

    initial_state = {
        "user_gender": user_input['gender'],
        "user_age_group": user_input['age'],
        "user_genres": user_input['genres'],
        "user_favorite_manga": user_input['favorite_manga'],
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

    print(f"\n최종 품질 점수: {final_state['recommendation_quality']*100:.0f}/100")

if __name__ == "__main__":
    asyncio.run(main())
