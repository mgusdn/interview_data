import re
import json
from pathlib import Path

DATA_DIR = Path("data/raw/ai-tech-interview/answers")
OUTPUT_PATH = Path("data/processed/qa.json")

CATEGORY_MAP = {
    "1-statistics-math.md": "statistics-math",
    "2-machine-learning.md": "machine-learning",
    "3-deep-learning.md": "deep-learning",
    "4-python.md": "python",
    "5-network.md": "network",
    "6-operating-system.md": "operating-system",
    "7-data-structure.md": "data-structure",
    "8-algorithm.md": "algorithm",
}


def parse_md_file(filepath: Path, category: str) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")

    lines = text.split("\n")
    qa_list = []
    current_question = None
    current_lines = []
    skip_references = False

    for line in lines:
        # #### References 시작 → 이후 내용 스킵 (질문 헤더보다 먼저 체크)
        if re.match(r"^####\s+References", line):
            skip_references = True

        # ### 또는 #### 질문 헤더 (파일마다 헤더 레벨이 다름)
        elif re.match(r"^#{3,4}\s+(.+?)$", line):
            # 이전 Q&A 저장
            if current_question:
                answer = "\n".join(current_lines).strip()
                if answer:
                    qa_list.append({
                        "category": category,
                        "question": current_question,
                        "answer": answer,
                        "source_file": filepath.name,
                    })
            current_question = re.match(r"^#{3,4}\s+(.+?)$", line).group(1).strip()
            current_lines = []
            skip_references = False

        # --- 구분선 → 스킵 리셋
        elif line.strip() == "---":
            skip_references = False

        # 본문 내용 수집
        else:
            if current_question and not skip_references:
                current_lines.append(line)

    # 마지막 Q&A 저장
    if current_question and current_lines:
        answer = "\n".join(current_lines).strip()
        if answer:
            qa_list.append({
                "category": category,
                "question": current_question,
                "answer": answer,
                "source_file": filepath.name,
            })

    return qa_list


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_qa = []

    for filename, category in CATEGORY_MAP.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"[SKIP] {filename} 파일 없음")
            continue

        qa_list = parse_md_file(filepath, category)
        print(f"[OK] {filename}: {len(qa_list)}개 Q&A 파싱됨")
        all_qa.extend(qa_list)

    OUTPUT_PATH.write_text(
        json.dumps(all_qa, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n총 {len(all_qa)}개 Q&A → {OUTPUT_PATH} 저장 완료")


if __name__ == "__main__":
    main()