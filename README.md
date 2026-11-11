# NCS-Physio AI

**Electrophysiological assessment of cervical pain, with LLM + RAG interpretation and a benchmark that evaluates AI reasoning on nerve conduction studies.**

Built from my MD (Physiology) thesis, *Electrophysiological Assessment of Cervical Pain: A Cross-sectional Study* (Rajiv Gandhi Medical College, Thane; guide: Dr. Swapnali Kadam). The study covers median, ulnar and radial NCS (latency, amplitude, conduction velocity, F-waves) in 150+ patients, with the Neck Disability Index, the IPAQ and body composition analysis.

> The public demo uses a **synthetic, physiology-based cohort**. No real patient data is included or should be uploaded.

## What it does

| Module | What it shows |
|---|---|
| **Patient entry & PDF report** | Structured NCS/NDI/IPAQ/body-composition capture, derived scores, an auditable rule-based interpretation, A4 PDF |
| **Rule engine** (`ncs_interpret.py`) | Localisation from physiology: myelin vs axon, pre- vs post-ganglionic (SNAP sparing in radiculopathy), focal vs generalised |
| **Synthetic cohort** (`ncs_synthetic.py`) | 150 cases generated from pathophysiology (e.g. F-wave latency scales with height; SNAPs spared in root lesions), each with a known ground-truth pattern |
| **Cohort analysis** (`ncs_analysis.py`) | Analysis-ready table; Spearman correlations, Mann-Whitney / Kruskal-Wallis, chi-square, Benjamini-Hochberg FDR; Plotly charts |
| **RAG** (`ncs_rag.py`) | BM25 implemented from scratch over a curated knowledge base (`kb_*.md`); the query is built from the case's abnormal findings |
| **LLM interpreter** (`ncs_ai.py`, `ncs_llm.py`) | Gemini / OpenAI / Claude over REST; JSON-structured output with citations |
| **AI critic** | Flags a wrong pattern or side, missed or invented abnormalities, **numbers absent from the data** (hallucination), invalid citations, and physiology contradictions |
| **Benchmark** (`ncs_benchmark.py`) | Stratified cases; RAG vs no-RAG ablation; rule-engine and always-normal baselines; confusion matrix |

## Run locally

```bash
git clone https://github.com/<your-username>/ncs-physio-ai.git
cd ncs-physio-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # 8 tests
mkdir .streamlit && cp secrets.toml.example .streamlit/secrets.toml   # then paste your API key
streamlit run app.py
```

To regenerate the demo cohort, run `python ncs_synthetic.py`.

## Deploy free on Streamlit Community Cloud

1. **Create a GitHub repository** named `ncs-physio-ai` (Public). Upload all the files in this folder (*Add file → Upload files*), then click **Commit changes**.
   Every file sits at the top level, so there are no folders to keep. Do **not** upload a `secrets.toml` file containing a real key.
2. **Get a free Gemini API key** at <https://aistudio.google.com> → *Get API key*.
3. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app → Deploy a public app from GitHub**.
   - Repository: `<your-username>/ncs-physio-ai`, branch `main`, main file `app.py`
   - Under *Advanced settings*, set Python 3.12 and paste the following into **Secrets**:
     ```toml
     GEMINI_API_KEY = "your-key"
     ```
4. Click **Deploy**. You get a URL like `https://ncs-physio-ai.streamlit.app` to put on your CV and LinkedIn.
5. Each commit to GitHub redeploys the app automatically.

Note: a public app's visitors use your API quota. The free Gemini tier is rate-limited, so keep benchmark runs small (12-24 cases), or remove the key after a demo.

### Alternative: Google Cloud Run
Cloud Run needs a billing account. With the `Dockerfile` in this repo:
```bash
gcloud run deploy ncs-physio-ai --source . --region asia-south1 --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your-key
```

## Methodological notes and limitations
- The reference limits are commonly cited adult textbook values. For research conclusions, replace them with laboratory or population-specific normative data, and adjust F-wave limits for height.
- On the synthetic set the rule engine scores 100% **by construction** (each case is accepted only when the rules reproduce its intended label). The set therefore tests whether an LLM can reach the transparent physiological answer; it does not test real-world diagnostic accuracy. The next step is to validate on de-identified real cases labelled by a neurophysiologist.
- The numeric-fidelity check is conservative: it can flag correctly derived numbers, such as a computed difference, which then need human review.
- This is a research and education tool, not a medical device.

## Project structure
All files sit at the top level of the repository, which makes uploading through the browser simple.
```
app.py                 entry point: registers the pages (st.navigation)
home.py, page_*.py     Streamlit pages (entry, reports, analysis, AI interpreter, benchmark, knowledge base)
ncs_*.py               core modules (no Streamlit dependency, unit-tested)
kb_*.md                RAG knowledge base
demo_cohort.csv        synthetic cohort
test_core.py           pytest suite
```
