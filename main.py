import argparse
import sys
from src.generate import generate_answer


def main():
    parser = argparse.ArgumentParser(
        description="AI 기술 면접 질문 RAG 시스템"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="검색할 질문 (예: 'TCP UDP 차이')"
    )

    args = parser.parse_args()

    print(f"\n질문: {args.query}")
    print("=" * 50)

    answer = generate_answer(args.query)
    print("\n")
    print("=" * 50)
    print("\n")
    print("답변:")
    print(answer)
    print("=" * 50)


if __name__ == "__main__":
    main()