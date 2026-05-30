# Evaluation Data

This folder holds QA pairs and manual scoring templates for Week 14.

## Files

- `qa_pairs.json`: Questions and expected sources used for retrieval evaluation.
- `answer_rubric.md`: Manual scoring sheet for answer quality.

## Usage

1. Crawl and index a target site.
2. Update `qa_pairs.json` with questions and expected URLs for that site.
3. Run retrieval evaluation:

```
python scripts/evaluate_retrieval.py --qa-file data/eval/qa_pairs.json
```

The report is written to `data/eval/retrieval_report.json`.