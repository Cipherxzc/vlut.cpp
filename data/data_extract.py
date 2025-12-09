from datasets import load_dataset
from pathlib import Path
import json



datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]


MIN_LENGTH_FOR_LONG = 500
MAX_LENGTH_FOR_LONG = 1000
MAX_WORDS_FOR_SHORT_ANSWER = 2

OUTPUT_PATH = Path("./data/longbench_en_longctx_shortans.jsonl")


def main():
    print(f"Output file  : {OUTPUT_PATH}")
    print(f"Long if {MIN_LENGTH_FOR_LONG} <= length <= {MAX_LENGTH_FOR_LONG}")
    print(f"Short answer if words <= {MAX_WORDS_FOR_SHORT_ANSWER}")
    print("-" * 60)

    if OUTPUT_PATH.exists():
        print(f"[Info] Remove existing file: {OUTPUT_PATH}")
        OUTPUT_PATH.unlink()


    total_kept = 0     

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for dataset in datasets:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            kept_this_dataset = 0

            for ex in data:
                if ex.get("language") != "en":
                    continue

                length = ex.get("length", 0)
                if length is None or length < MIN_LENGTH_FOR_LONG or length > MAX_LENGTH_FOR_LONG:
                    continue

                answers = ex.get("answers", [])
                if answers is None or len(str(answers[0]).strip()) <= MAX_WORDS_FOR_SHORT_ANSWER:
                    continue

                record = {
                    "_id": ex.get("_id"),
                    "dataset": ex.get("dataset", dataset),
                    "language": ex.get("language"),
                    "length": length,
                    "input": ex.get("input"),
                    "context": ex.get("context"),
                    "answers": answers,
                    "all_classes": ex.get("all_classes"),
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_this_dataset += 1
                total_kept += 1
            
            print(f"  -> kept {kept_this_dataset} samples for config '{dataset}'")
            print("-" * 60)

    print(f"Done. Total kept samples: {total_kept}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")




if __name__ == "__main__":
    main()