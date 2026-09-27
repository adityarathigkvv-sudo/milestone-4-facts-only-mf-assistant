# Milestone 4 — Facts-Only MF Assistant

## Product selected
**Groww**

The same product is intended to be used across the milestone series.

## Prototype scope
This prototype is a **facts-only RAG assistant** for:
- SBI Large Cap Fund (previously SBI BlueChip Fund)
- SBI Flexicap Fund
- SBI ELSS Tax Saver Fund

The corpus uses official public sources from:
- SBI Mutual Fund
- AMFI
- SEBI

## What it answers
Examples:
- Expense ratio (TER) facts as of 31 Mar 2026
- Expense/load facts when present in the corpus
- Exit load
- Minimum SIP / minimum investment
- ELSS lock-in
- Riskometer
- Benchmark
- How to access/download statements

## What it refuses
The assistant refuses:
- Buy/sell/hold recommendations
- “Which fund is best?” questions
- Portfolio allocation/suitability questions
- Return/performance comparison requests
- Requests involving PAN, Aadhaar, OTPs, bank/account numbers, phone numbers or email addresses

## RAG architecture
1. Official-source facts are stored as small retrievable chunks in `data/corpus.json`.
2. TF-IDF + cosine similarity retrieves the closest chunks.
3. Topic-aware routing improves precision for common FAQ intents.
4. A constrained answer template returns the fact and exactly one official source link.
5. Safety rules run before retrieval.

## Files
- `app.py` — Streamlit working prototype
- `data/corpus.json` — retrieval corpus
- `data/sources.csv` — 18 official public source URLs
- `prompt.txt` — LLM/system-prompt specification used for the design
- `sample_qa.md` — sample test questions and expected answers
- `disclaimer.md` — UI disclaimer text
- `requirements.txt` — Python dependencies

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The terminal will show a local URL such as `http://localhost:8501`.

## Hosting on Streamlit Community Cloud
1. Create a public GitHub repository, e.g. `milestone-3-facts-only-mf-assistant`.
2. Upload all files in this folder, keeping `data/corpus.json` and `data/sources.csv` in the `data/` folder.
3. Go to Streamlit Community Cloud and create a new app.
4. Select the GitHub repository and `app.py`.
5. Deploy.
6. Open the deployed URL in an incognito/private browser and test it before submission.

## Submission checklist
### GitHub
Share the repository URL. The README already contains:
- scope
- setup
- architecture
- safety constraints
- source list reference
- sample Q&A reference
- disclaimer reference

### Working prototype
Share the deployed Streamlit URL.

### Demo video
Only needed if the prototype is not hosted. Keep it <=3 minutes and show:
1. Welcome screen + disclaimer
2. Factual question with a citation
3. Second factual question
4. Advice/refusal example
5. PII refusal example
6. Brief view of GitHub README/source list

## Important limitation
The prototype intentionally uses a **small, curated corpus** rather than crawling the web live. It therefore does not claim to know facts outside the listed sources. For production use, the corpus would need a scheduled source-refresh pipeline and stronger document-level retrieval/citation validation.
