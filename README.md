# Smart Resume Analyzer

A Resume Parsing and Scoring application designed to simulate modern Applicant Tracking Systems (ATS). This tool evaluates the "fit" between a candidate's resume and a job description by analyzing three distinct metrics: semantic meaning, hard skills, and vocabulary usage.

## Overview
Traditional ATS systems often reject qualified candidates because they lack specific keywords, even if their experience is relevant. Conversely, they may rank candidates highly just for "keyword stuffing."

This project solves that by implementing a **Hybrid Search** approach. It doesn't just count words; it uses AI (Sentence Transformers) to understand the *context* of the experience, while still verifying mandatory technical requirements.

## Key Features
- **Semantic Matching:** Uses the `all-MiniLM-L6-v2` transformer model to detect conceptual similarity (e.g., understanding that "Canteen Staff" implies "Customer Service").
- **Skill Gap Analysis:** Identifies missing technical skills required by the Job Description.
- **Dynamic Scoring:** Calculates a weighted percentage based on three distinct metrics.
- **Interactive UI:** Built with Streamlit to provide real-time feedback and visual score breakdowns.

## The Scoring Logic (The "Golden Ratio")
The system calculates a final match score based on the following weighted formula:

| Metric | Weight | Purpose |
| :--- | :--- | :--- |
| **Semantic Match (SBERT)** | **60%** | The "Vibe Check" Measures how well the *meaning* of the resume aligns with the JD. This captures seniority and domain fit. |
| **Hard Skills (Keywords)** | **30%** | The "Gatekeeper." Checks for the presence of mandatory technical skills (e.g., Python, SQL) defined in the ontology. |
| **Vocabulary (TF-IDF)** | **10%** | The "Bonus." Checks if the candidate uses standard industry terminology similar to the JD. |

**Why this split?**
We prioritize Semantic Matching (60%) because modern recruitment focuses on transferable experience. Hard skills (30%) remain critical for technical roles, while vocabulary (10%) is weighted low to prevent "keyword stuffing" from skewing the results.

## Tech Stack
- **Python 3.10+**
- **Streamlit:** Frontend interface.
- **Sentence-Transformers (SBERT):** For vector embeddings and semantic search.
- **Scikit-Learn:** For TF-IDF vectorization and Cosine Similarity.
- **NLTK:** For text preprocessing (Lemmatization, Stopword removal).
- **Plotly:** For interactive data visualization.

## Setup & Installation

1. **Clone the repository:**
   ```bash
    git clone https://github.com/HarshVishwakarma20/AI-Resume-Analyzer.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
4. **Open in browser:** Navigate to `http://localhost:8501`

## Limitations

While functional, this system has specific limitations typical of rule-based/hybrid systems:

1. **Static Skills Ontology:** The current version uses a hardcoded dictionary (skills_db) to detect skills. It does not automatically discover new technologies.

2. **PDF-Only Support:** Currently, the parser is optimized for standard PDF text layers. It does not perform OCR on scanned image-based resumes.

3. **Processing Speed:** The SBERT model runs locally on the CPU. While accurate, it may take a few seconds to compute embeddings for long documents on lower-end hardware.

## Future Improvements

**NER Integration:** Replace the dictionary-based skill extractor with a Named Entity Recognition model (like spaCy) to automatically detect Org, Tech, and GPE entities.

**OCR Integration:** Add pytesseract to handle image-based PDFs.

**Cloud Deployment:** optimize memory usage for deployment on free-tier cloud instances.

## License

This project is open-source and available for educational purposes.
