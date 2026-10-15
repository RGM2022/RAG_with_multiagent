# Local Vector RAG

A local Streamlit document-search app that uses Hugging Face text and visual
embedding models with persistent ChromaDB storage for retrieval. Answers are
source passages or matching images, not LLM-generated text. Documents are
loaded and chunked with the existing `RAG/search_engine.py` implementation.

## Run on Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
from this folder run:

```powershell
uv sync
uv run streamlit run app.py
```

The first indexing or search operation downloads the required text or visual
embedding model from Hugging Face. The models are public and normally do not
require a token. If your Hugging Face setup requires authentication, set `HF_TOKEN` in PowerShell
before starting Streamlit:

```powershell
$env:HF_TOKEN = "your-new-token"
uv run streamlit run app.py
```

Do not commit tokens or put them in source files. The Chroma database is
persisted locally in `chroma_db/`, which is excluded from version control.

Upload supported TXT, Markdown, CSV, JSON, or text-based PDF files; select
reusable files under `knowledge_base`; scrape public webpages; or enter direct
image URLs. Scraped webpages also contribute up to 20 linked raster images per
page. JPEG, PNG, WebP, GIF, BMP, and TIFF images are supported; images are
validated, capped at 8 MB and 20 megapixels each. Choose **Index selected documents** to embed
and save the currently selected sources. Text uses
`sentence-transformers/all-MiniLM-L6-v2`; images and text search queries use
the shared visual/text space from `sentence-transformers/clip-ViT-B-32`.
Image vectors are stored in a separate Chroma collection so they do not mix
with existing text passage vectors.

Use **Search indexed documents** to retrieve semantically similar passages
and visually similar images from all sources saved in ChromaDB, including
sources indexed in earlier sessions. The source selector can narrow a search
to chosen indexed sources, and the default minimum cosine similarity of `0.35`
hides weak matches; adjust the slider when you want broader or stricter
results. If no passages or images meet the threshold, the app shows the closest
matches separately and labels them as low-confidence. Indexing a source again
replaces its previous passages. PDF page downloads are available when the
matching PDF is currently selected or uploaded. Scanned PDFs are not OCR
processed.

Passages are limited to 150 lexical tokens with 30-token overlap. This
reduces truncation by the embedding model, whose model card documents that
inputs longer than 256 word pieces are truncated. This version stores vectors
in a new Chroma collection; reselect/re-upload local files and scrape webpages
again, then choose **Index selected documents** to create fresh vectors. The
previous collection is left untouched.

Visual image vectors use a separate collection and do not require re-indexing
existing text sources.

This folder imports the existing ingestion and chunking functions from its
sibling `RAG/` folder, so keep both folders together.
