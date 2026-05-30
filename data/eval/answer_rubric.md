# Answer Quality Rubric

Use this rubric to score answers produced by the system. Scores are 1-5.

## Scoring Criteria

1. **Correctness**
   - 5: Completely correct and directly answers the question.
   - 3: Partially correct with minor errors or omissions.
   - 1: Incorrect or irrelevant.

2. **Grounding / Citations**
   - 5: Claims are supported by sources; citations match evidence.
   - 3: Some claims are grounded, but citations are weak or incomplete.
   - 1: No meaningful grounding or citations are incorrect.

3. **Completeness**
   - 5: Covers all essential points expected from the sources.
   - 3: Covers some points but misses key information.
   - 1: Misses most of the expected content.

4. **Clarity**
   - 5: Clear, concise, easy to understand.
   - 3: Understandable but verbose or slightly confusing.
   - 1: Hard to understand.

5. **Refusal Quality (if refused)**
   - 5: Correctly refuses when evidence is insufficient.
   - 3: Refusal is acceptable but could be more precise.
   - 1: Refuses when evidence exists or answers when it should refuse.

## Manual Scoring Sheet

Fill one row per question.

| ID | Question | Expected URLs/Notes | Answer Summary | Correctness (1-5) | Grounding (1-5) | Completeness (1-5) | Clarity (1-5) | Refusal Quality (1-5) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| example-1 | What is this domain used for? | https://example.com |  |  |  |  |  |  |  |
| example-2 | Why is this domain reserved? | https://example.com |  |  |  |  |  |  |  |
| example-3 | Where can I learn more about this domain? | https://example.com |  |  |  |  |  |  |  |
| site-1 | What is the main purpose of the site? | Update QA file |  |  |  |  |  |  |  |
| site-2 | What topics are covered on the about page? | Update QA file |  |  |  |  |  |  |  |
