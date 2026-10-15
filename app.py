from pathlib import Path
import os
import sys
from dataclasses import replace

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RAG.search_engine import (  # noqa: E402
    ImageAsset,
    SearchResult,
    SUPPORTED_SUFFIXES,
    chunk_documents,
    extract_pdf_page,
    fetch_page_images,
    load_document,
    search as rank_passages,
    scrape_website,
)
from vector_store import (  # noqa: E402
    IMAGE_MIN_SIMILARITY,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_CHUNK_SIZE,
    ChromaVectorStore,
    ImageTextEmbedder,
    TextEmbedder,
    image_filename_matches_query,
    load_embedder,
    load_image_embedder,
)


KNOWLEDGE_BASE_DIR = APP_DIR / "knowledge_base"
CHROMA_DIR = APP_DIR / "chroma_db"

st.set_page_config(page_title="Local Vector RAG", page_icon="🔎", layout="wide")
st.title("Local Vector RAG")
st.caption(
    "Semantic text and image search with Hugging Face embeddings and a persistent local ChromaDB."
)

with st.expander("How it works"):
    st.write(
        "Selected documents are split into overlapping passages, embedded with "
        "all-MiniLM-L6-v2. Images and text search queries use CLIP visual embeddings. "
        "Both are stored in local ChromaDB collections and retrieved by vector "
        "similarity; results link to indexed content rather than generating an answer."
    )
    st.write(
        "Supported files: TXT, Markdown, CSV, JSON, and text-based PDF. "
        "Scanned PDFs are not OCR processed."
    )

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
uploaded_files = st.file_uploader(
    "Upload documents",
    type=[suffix.removeprefix(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
    accept_multiple_files=True,
    help="Uploaded files are available for indexing in this session.",
)
knowledge_base_files = sorted(
    (
        file_path
        for file_path in KNOWLEDGE_BASE_DIR.rglob("*")
        if file_path.is_file() and file_path.suffix.casefold() in SUPPORTED_SUFFIXES
    ),
    key=lambda file_path: file_path.relative_to(KNOWLEDGE_BASE_DIR).as_posix().casefold(),
)
knowledge_base_sources = {
    file_path.relative_to(KNOWLEDGE_BASE_DIR).as_posix(): file_path
    for file_path in knowledge_base_files
}
selected_knowledge_base_sources = st.multiselect(
    "Select reusable documents from knowledge_base",
    options=list(knowledge_base_sources),
)

st.subheader("Scrape a website")
st.caption(
    "Enter one public webpage URL per line. Readable text and up to 20 linked "
    "raster images per page will be fetched for multimodal indexing."
)
with st.form("website_form"):
    source_urls_input = st.text_area(
        "Website URLs",
        key="website-urls",
        placeholder="https://example.com/document\nhttps://example.org/another-page",
    )
    scrape_submitted = st.form_submit_button("Scrape webpages")

if "scraped_documents" not in st.session_state:
    st.session_state["scraped_documents"] = []
if "scraped_images" not in st.session_state:
    st.session_state["scraped_images"] = []

if scrape_submitted:
    source_urls = list(dict.fromkeys(
        url.strip()
        for url in source_urls_input.splitlines()
        if url.strip()
    ))
    if not source_urls:
        st.error("Enter at least one website URL to scrape.")
    else:
        scraped_by_url = {
            document.source_url: document
            for document in st.session_state["scraped_documents"]
        }
        images_by_reference = {
            (image.image_url, image.page_url): image
            for image in st.session_state["scraped_images"]
        }
        for source_url in source_urls:
            try:
                document = scrape_website(source_url)
                scraped_by_url[document.source_url] = document
                st.success(f"Scraped {document.source}")
                if document.image_references:
                    with st.status(
                        f"Fetching {len(document.image_references)} page image(s) concurrently...",
                        expanded=False,
                    ) as image_status:
                        images, image_errors = fetch_page_images(
                            document.source_url,
                            document.image_references,
                        )
                        for image in images:
                            images_by_reference[(image.image_url, image.page_url)] = image
                        for image_reference, error in image_errors:
                            st.error(
                                f"Could not fetch {image_reference.image_url}: {error}"
                            )
                        image_status.update(
                            label=(
                                f"Fetched {len(images)} of "
                                f"{len(document.image_references)} page image(s)"
                            ),
                            state="complete",
                        )
            except ValueError as error:
                st.error(f"{source_url}: {error}")
        st.session_state["scraped_documents"] = list(scraped_by_url.values())
        st.session_state["scraped_images"] = list(images_by_reference.values())

scraped_documents = st.session_state["scraped_documents"]
if scraped_documents:
    st.caption("Webpages with text and images ready to index:")
    for document in scraped_documents:
        image_count = sum(
            image.page_url == document.source_url
            for image in st.session_state["scraped_images"]
        )
        st.markdown(
            f"- [{document.source}]({document.source_url}) "
            f"— text and {image_count} image(s)"
        )
    if st.button("Clear scraped webpages"):
        st.session_state["scraped_documents"] = []
        st.session_state["scraped_images"] = []
        st.rerun()

scraped_images: list[ImageAsset] = st.session_state["scraped_images"]

index_batches: list[tuple[str, str, list[SearchResult]]] = []
image_batches: dict[str, list[ImageAsset]] = {}
pdf_source_contents: dict[str, bytes] = {}
load_errors: list[str] = []
document_page_count = 0

for source in selected_knowledge_base_sources:
    file_path = knowledge_base_sources[source]
    try:
        content = file_path.read_bytes()
        documents = load_document(source, content)
        passages = chunk_documents(
            documents,
            chunk_size=VECTOR_CHUNK_SIZE,
            overlap=VECTOR_CHUNK_OVERLAP,
        )
        index_batches.append((f"knowledge_base:{source}", source, passages))
        document_page_count += len(documents)
        if file_path.suffix.casefold() == ".pdf":
            pdf_source_contents[source] = content
    except (OSError, ValueError, ImportError) as error:
        load_errors.append(f"{source}: {error}")

for uploaded_file in uploaded_files or []:
    try:
        content = uploaded_file.getvalue()
        documents = load_document(uploaded_file.name, content)
        passages = chunk_documents(
            documents,
            chunk_size=VECTOR_CHUNK_SIZE,
            overlap=VECTOR_CHUNK_OVERLAP,
        )
        index_batches.append((f"upload:{uploaded_file.name}", uploaded_file.name, passages))
        document_page_count += len(documents)
        if Path(uploaded_file.name).suffix.casefold() == ".pdf":
            pdf_source_contents[uploaded_file.name] = content
    except (OSError, ValueError, ImportError) as error:
        load_errors.append(f"{uploaded_file.name}: {error}")

for document in scraped_documents:
    passages = chunk_documents(
        [document],
        chunk_size=VECTOR_CHUNK_SIZE,
        overlap=VECTOR_CHUNK_OVERLAP,
    )
    if document.source_url:
        passages = [
            replace(passage, source=document.source_url)
            for passage in passages
        ]
    index_batches.append((f"website:{document.source_url}", document.source, passages))
    document_page_count += 1

for image in scraped_images:
    if image.page_url is None:
        st.error("Standalone images are no longer supported. Scrape the webpage URL instead.")
        continue
    source_id = f"website-images:{image.page_url}"
    image_batches.setdefault(source_id, []).append(image)

for error in load_errors:
    st.error(error)

st.caption(
    f"{document_page_count} document page(s) currently selected; "
    f"{sum(len(passages) for _, _, passages in index_batches)} passage(s) and "
    f"{sum(len(images) for images in image_batches.values())} image(s) ready to index."
)
@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(CHROMA_DIR)


@st.cache_resource
def get_embedder(token: str | None) -> TextEmbedder:
    return load_embedder(token)


@st.cache_resource
def get_image_embedder(token: str | None) -> ImageTextEmbedder:
    return load_image_embedder(token)


vector_store = get_vector_store()
if st.button("Index selected documents", type="primary"):
    if not index_batches and not image_batches:
        st.error("Select or upload documents, scrape a webpage, or fetch an image before indexing.")
    else:
        token = os.environ.get("HF_TOKEN") or None
        indexed_passages = 0
        if index_batches:
            embedder = get_embedder(token)
            for source_id, source, passages in index_batches:
                indexed_passages += vector_store.index_source(source_id, passages, embedder)
                st.write(f"Indexed text from {source}")
        indexed_images = 0
        if image_batches:
            image_embedder = get_image_embedder(token)
            for source_id, images in image_batches.items():
                try:
                    indexed_images += vector_store.index_images(
                        source_id,
                        images,
                        image_embedder,
                    )
                except (OSError, ValueError, ImportError) as error:
                    st.error(f"Could not index images from {source_id}: {error}")
                else:
                    st.write(f"Indexed {len(images)} image(s) from {source_id}")
        st.success(
            f"Saved {indexed_passages} text passage(s) and {indexed_images} image(s) "
            "to the local ChromaDB."
        )

st.caption(
    f"Persistent ChromaDB: {vector_store.count()} text passage(s) and "
    f"{vector_store.image_count()} image(s) across {len(vector_store.list_sources())} source(s)."
)
available_sources = vector_store.list_sources()
available_image_sources = set(vector_store.list_image_sources())
available_text_sources = set(vector_store.list_text_sources())
selected_search_sources = st.multiselect(
    "Search within indexed sources",
    options=available_sources,
    default=available_sources,
    help="By default, searches all indexed sources. Uncheck unrelated sources to narrow results.",
)
st.caption(
    "Text passages are ranked by keyword relevance across the selected sources. "
    f"Image search uses a {IMAGE_MIN_SIMILARITY:.2f} similarity threshold."
)

with st.form("question_form"):
    query = st.text_input(
        "Ask a question about your indexed documents",
        placeholder="e.g. What does debugging mean?",
    )
    submitted = st.form_submit_button("Search indexed documents")

if submitted and query.strip():
    if not available_sources:
        st.info("Index at least one document before searching.")
    elif not selected_search_sources:
        st.info("Select at least one indexed source to search.")
    else:
        selected_sources = set(selected_search_sources)
        text_results = []
        image_results = []
        if selected_sources & available_text_sources:
            embedder = get_embedder(os.environ.get("HF_TOKEN") or None)
            text_candidates = vector_store.search(
                query.strip(),
                embedder,
                top_k=min(100, vector_store.count()),
                min_similarity=-1.0,
                sources=selected_search_sources,
            )
            candidate_by_identity = {
                (result.source, result.page, result.chunk_id): result
                for result in text_candidates
            }
            ranked_passages = rank_passages(
                query.strip(),
                [
                    SearchResult(
                        source=result.source,
                        page=result.page,
                        chunk_id=result.chunk_id,
                        text=result.text,
                        score=result.similarity,
                        source_url=result.source_url,
                    )
                    for result in text_candidates
                ],
                top_k=5,
            )
            if ranked_passages:
                best_similarity = max(
                    result.similarity for result in text_candidates
                )
                ranked_passages = [
                    passage
                    for passage in ranked_passages
                    if candidate_by_identity[
                        (passage.source, passage.page, passage.chunk_id)
                    ].similarity
                    >= best_similarity * 0.85
                ]
            text_results = [
                candidate_by_identity[
                    (passage.source, passage.page, passage.chunk_id)
                ]
                for passage in ranked_passages
            ]
            text_results = vector_store.merge_adjacent_passages(text_results)[:2]
        if text_results:
            matching_image_sources = {
                result.source_url.partition("#")[0]
                if result.source_url
                else result.source
                for result in text_results
            } & available_image_sources
        else:
            matching_image_sources = selected_sources & available_image_sources
        if matching_image_sources:
            image_embedder = get_image_embedder(os.environ.get("HF_TOKEN") or None)
            image_candidates = vector_store.search_images(
                query.strip(),
                image_embedder,
                top_k=min(50, vector_store.image_count()),
                min_similarity=-1.0,
                sources=matching_image_sources,
            )
            matching_text = " ".join(result.text for result in text_results)
            named_images = [
                result
                for result in image_candidates
                if image_filename_matches_query(
                    result.image_url,
                    query.strip(),
                    matching_text,
                )
            ]
            image_results = named_images[:1] or [
                result
                for result in image_candidates
                if result.similarity >= IMAGE_MIN_SIMILARITY
            ][:1]
        if not text_results and not image_results:
            st.info("No relevant information found in the selected sources.")
        if text_results:
            st.subheader("Relevant passages")
            for result in text_results:
                st.markdown(
                    f"**{result.citation}** · cosine similarity {result.similarity:.3f}"
                )
                if result.source_url:
                    st.markdown(f"[Open source URL]({result.source_url})")
                st.write(result.text)
                if result.page is not None and result.source in pdf_source_contents:
                    try:
                        page_pdf = extract_pdf_page(
                            pdf_source_contents[result.source],
                            result.page,
                        )
                    except (OSError, ValueError, ImportError) as error:
                        st.error(f"Could not prepare the matching PDF page: {error}")
                    else:
                        st.download_button(
                            f"Open PDF page {result.page} (passage {result.chunk_id})",
                            page_pdf,
                            file_name=f"{Path(result.source).stem}-page-{result.page}.pdf",
                            mime="application/pdf",
                            key=f"pdf-page-{result.source}-{result.page}-{result.chunk_id}",
                        )
        if image_results:
            st.subheader("Relevant images and diagrams")
            for result in image_results:
                st.markdown(
                    f"**Visual match** · cosine similarity {result.similarity:.3f}"
                )
                st.image(
                    result.image_url,
                    caption=result.alt_text or result.image_url,
                )
                if result.page_url:
                    st.markdown(f"[Open source webpage]({result.page_url})")
                st.markdown(f"[Open original image]({result.image_url})")
