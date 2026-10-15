"""Persistent ChromaDB storage and semantic retrieval for document passages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from math import isfinite
from pathlib import Path
import re
from typing import Any, Iterable, Protocol
from urllib.parse import unquote, urlsplit

from RAG.search_engine import MAX_IMAGE_PIXELS, ImageAsset, SearchResult


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "local_rag_passages_v2"
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
IMAGE_COLLECTION_NAME = "local_rag_images_v1"
IMAGE_EMBEDDING_BATCH_SIZE = 2
IMAGE_MIN_SIMILARITY = 0.25
VECTOR_CHUNK_SIZE = 150
VECTOR_CHUNK_OVERLAP = 30


class TextEmbedder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> Any: ...


class ImageTextEmbedder(Protocol):
    def encode(self, inputs: list[Any], *, normalize_embeddings: bool) -> Any: ...


@dataclass(frozen=True)
class VectorSearchResult:
    source: str
    page: int | None
    chunk_id: int
    text: str
    similarity: float
    source_url: str | None

    @property
    def citation(self) -> str:
        location = f", page {self.page}" if self.page is not None else ""
        return f"{self.source}{location}, passage {self.chunk_id}"


@dataclass(frozen=True)
class VectorImageResult:
    source: str
    image_url: str
    page_url: str | None
    alt_text: str
    similarity: float


class ChromaVectorStore:
    def __init__(self, persist_directory: Path) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._image_collection = self._client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        sources = set(self.list_text_sources())
        sources.update(self.list_image_sources())
        return sorted(sources, key=str.casefold)

    def list_text_sources(self) -> list[str]:
        return sorted(self._collection_sources(self._collection), key=str.casefold)

    def list_image_sources(self) -> list[str]:
        return sorted(self._collection_sources(self._image_collection), key=str.casefold)

    def image_count(self) -> int:
        return self._image_collection.count()

    def index_source(
        self,
        source_id: str,
        passages: Iterable[SearchResult],
        embedder: TextEmbedder,
    ) -> int:
        passages = list(passages)
        if not passages:
            return 0

        ids = [
            _passage_id(source_id, passage)
            for passage in passages
        ]
        texts = [passage.text for passage in passages]
        embeddings = _as_embedding_list(
            embedder.encode(texts, normalize_embeddings=True)
        )
        if len(embeddings) != len(passages):
            raise ValueError("The embedding model returned an unexpected number of vectors.")

        metadatas = [
            _passage_metadata(source_id, passage)
            for passage in passages
        ]
        existing = self._collection.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stale_ids = sorted(set(existing["ids"]) - set(ids))
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        return len(passages)

    def index_images(
        self,
        source_id: str,
        images: Iterable[ImageAsset],
        embedder: ImageTextEmbedder,
    ) -> int:
        images = list(images)
        if not images:
            return 0

        from PIL import Image

        embeddings = []
        for start in range(0, len(images), IMAGE_EMBEDDING_BATCH_SIZE):
            decoded_images = []
            try:
                for asset in images[start : start + IMAGE_EMBEDDING_BATCH_SIZE]:
                    with Image.open(BytesIO(asset.content)) as image:
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise ValueError("An indexed image exceeds the 20-megapixel limit.")
                        decoded_images.append(image.convert("RGB"))
                batch_embeddings = _as_embedding_list(
                    embedder.encode(decoded_images, normalize_embeddings=True)
                )
                if len(batch_embeddings) != len(decoded_images):
                    raise ValueError(
                        "The visual embedding model returned an unexpected number of vectors."
                    )
                embeddings.extend(batch_embeddings)
            finally:
                for image in decoded_images:
                    image.close()
        if len(embeddings) != len(images):
            raise ValueError("The visual embedding model returned an unexpected number of vectors.")

        ids = [_image_id(source_id, image) for image in images]
        documents = [image.alt_text or image.image_url for image in images]
        metadatas = [_image_metadata(source_id, image) for image in images]
        existing = self._image_collection.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        self._image_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stale_ids = sorted(set(existing["ids"]) - set(ids))
        if stale_ids:
            self._image_collection.delete(ids=stale_ids)
        return len(images)

    def search(
        self,
        query: str,
        embedder: TextEmbedder,
        top_k: int = 5,
        min_similarity: float = 0.35,
        sources: Iterable[str] | None = None,
    ) -> list[VectorSearchResult]:
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1.")
        if top_k <= 0 or self.count() == 0:
            return []

        source_filter = None if sources is None else sorted(set(sources), key=str.casefold)
        if source_filter == []:
            return []

        query_embedding = _as_embedding_list(
            embedder.encode([query], normalize_embeddings=True)
        )
        if len(query_embedding) != 1:
            raise ValueError("The embedding model returned an unexpected query vector.")
        query_options: dict[str, Any] = {}
        if source_filter is not None:
            page_passages = self._collection.get(
                where={"source_url": {"$in": source_filter}},
                include=["metadatas"],
            )
            source_filter = sorted(
                set(source_filter)
                | {
                    metadata["source"]
                    for metadata in page_passages["metadatas"]
                    if metadata and "source" in metadata
                },
                key=str.casefold,
            )
            query_options["where"] = {"source": {"$in": source_filter}}
        matches = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
            **query_options,
        )
        ids = matches["ids"][0]
        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise RuntimeError("ChromaDB returned mismatched passage search results.")

        results: list[VectorSearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            if text is None or metadata is None:
                raise RuntimeError("ChromaDB returned a passage without its stored content or metadata.")
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            results.append(
                VectorSearchResult(
                    source=metadata["source"],
                    page=metadata.get("page"),
                    chunk_id=metadata["chunk_id"],
                    text=text,
                    similarity=similarity,
                    source_url=metadata.get("source_url"),
                )
            )
        return results

    def merge_adjacent_passages(
        self,
        results: Iterable[VectorSearchResult],
        max_chunk_gap: int = 4,
    ) -> list[VectorSearchResult]:
        results = list(results)
        if max_chunk_gap < 0:
            raise ValueError("max_chunk_gap must be non-negative.")

        groups: list[list[VectorSearchResult]] = []
        for result in results:
            matching_group = None
            for group in groups:
                chunk_ids = [item.chunk_id for item in group]
                if (
                    group[0].source == result.source
                    and group[0].source_url == result.source_url
                    and max(max(chunk_ids), result.chunk_id)
                    - min(min(chunk_ids), result.chunk_id)
                    <= max_chunk_gap
                ):
                    matching_group = group
                    break
            if matching_group is None:
                groups.append([result])
            else:
                matching_group.append(result)

        merged_results: list[VectorSearchResult] = []
        for group in groups:
            best_match = max(group, key=lambda result: result.similarity)
            heading_only = (
                best_match.source_url is not None
                and len(best_match.text.strip()) <= 80
                and not re.search(r"[.!?]", best_match.text)
            )
            expand_section = len(group) == 1 and not heading_only
            if best_match.source_url is None:
                merged_results.append(best_match)
                continue

            first_chunk = min(result.chunk_id for result in group)
            last_chunk = max(result.chunk_id for result in group)
            if heading_only and len(group) == 1:
                last_chunk = first_chunk + max(0, max_chunk_gap - 1)
            elif expand_section:
                first_chunk = max(0, first_chunk - max_chunk_gap)
                last_chunk += max_chunk_gap
            base_page_url = best_match.source_url.partition("#")[0]
            stored = self._collection.get(
                where={"source": {"$in": [best_match.source]}},
                include=["documents", "metadatas"],
            )
            context_passages = sorted(
                (
                    (metadata["chunk_id"], text)
                    for text, metadata in zip(
                        stored["documents"],
                        stored["metadatas"],
                    )
                    if metadata
                    and metadata.get("source_url", "").partition("#")[0]
                    == base_page_url
                    and (
                        not expand_section
                        or (
                            metadata.get("source_url") == best_match.source_url
                            and abs(metadata.get("chunk_id", -1) - best_match.chunk_id)
                            <= max_chunk_gap
                        )
                    )
                    and first_chunk <= metadata.get("chunk_id", -1) <= last_chunk
                    and text is not None
                ),
                key=lambda item: item[0],
            )
            if not context_passages:
                merged_results.append(best_match)
                continue

            merged_results.append(
                VectorSearchResult(
                    source=best_match.source,
                    page=best_match.page,
                    chunk_id=first_chunk,
                    text="\n\n".join(text for _, text in context_passages),
                    similarity=best_match.similarity,
                    source_url=best_match.source_url,
                )
            )

        return sorted(
            merged_results,
            key=lambda result: result.similarity,
            reverse=True,
        )

    def search_images(
        self,
        query: str,
        embedder: ImageTextEmbedder,
        top_k: int = 5,
        min_similarity: float = 0.35,
        sources: Iterable[str] | None = None,
    ) -> list[VectorImageResult]:
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1.")
        image_count = self.image_count()
        if top_k <= 0 or image_count == 0:
            return []

        source_filter = None if sources is None else sorted(set(sources), key=str.casefold)
        if source_filter == []:
            return []

        query_embedding = _as_embedding_list(
            embedder.encode([query], normalize_embeddings=True)
        )
        if len(query_embedding) != 1:
            raise ValueError("The visual embedding model returned an unexpected query vector.")
        query_options: dict[str, Any] = {}
        if source_filter is not None:
            query_options["where"] = {"source": {"$in": source_filter}}
        matches = self._image_collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, image_count),
            include=["metadatas", "distances"],
            **query_options,
        )
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]
        if len(metadatas) != len(distances):
            raise RuntimeError("ChromaDB returned mismatched image search results.")

        results: list[VectorImageResult] = []
        for metadata, distance in zip(metadatas, distances):
            if metadata is None:
                raise RuntimeError("ChromaDB returned an image without its stored metadata.")
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            results.append(
                VectorImageResult(
                    source=metadata["source"],
                    image_url=metadata["image_url"],
                    page_url=metadata.get("page_url"),
                    alt_text=metadata.get("alt_text", ""),
                    similarity=similarity,
                )
            )
        return results

    @staticmethod
    def _collection_sources(collection: Any) -> set[str]:
        stored = collection.get(include=["metadatas"])
        return {
            metadata["source"]
            for metadata in stored["metadatas"]
            if metadata and "source" in metadata
        }


def load_embedder(token: str | None = None) -> TextEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, token=token)


def load_image_embedder(token: str | None = None) -> ImageTextEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(IMAGE_MODEL_NAME, token=token)


def _as_embedding_list(embeddings: Any) -> list[list[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    vectors = [[float(value) for value in vector] for vector in embeddings]
    for vector in vectors:
        if not vector or not all(map(isfinite, vector)):
            raise ValueError("The embedding model returned an empty or non-finite vector.")
    return vectors


def _passage_id(source_id: str, passage: SearchResult) -> str:
    identity = "\x1f".join(
        (
            source_id,
            str(passage.page or ""),
            str(passage.chunk_id),
            passage.text,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _passage_metadata(source_id: str, passage: SearchResult) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "source_id": source_id,
        "source": passage.source,
        "chunk_id": passage.chunk_id,
    }
    if passage.page is not None:
        metadata["page"] = passage.page
    if passage.source_url is not None:
        metadata["source_url"] = passage.source_url
    return metadata


def _image_id(source_id: str, image: ImageAsset) -> str:
    identity = "\x1f".join((source_id, image.image_url))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _image_metadata(source_id: str, image: ImageAsset) -> dict[str, str]:
    metadata = {
        "source_id": source_id,
        "source": image.page_url or image.image_url,
        "image_url": image.image_url,
        "alt_text": image.alt_text,
    }
    if image.page_url is not None:
        metadata["page_url"] = image.page_url
    return metadata


def image_filename_matches_text(image_url: str, text: str) -> bool:
    filename = unquote(urlsplit(image_url).path.rsplit("/", 1)[-1])
    filename = re.sub(r"[^a-z0-9]", "", Path(filename).stem.casefold())
    normalized_text = re.sub(r"[^a-z0-9]", "", text.casefold())
    if len(filename) >= 4 and filename in normalized_text:
        return True
    if len(filename) < 3:
        return False

    words = re.findall(r"[a-z0-9]+", text.casefold())
    if filename in words:
        return True
    acronym_width = len(filename)
    return any(
        "".join(word[0] for word in words[start : start + acronym_width])
        == filename
        for start in range(len(words) - acronym_width + 1)
    )


def image_filename_matches_query(
    image_url: str,
    query: str,
    matching_text: str,
) -> bool:
    return image_filename_matches_text(
        image_url,
        query,
    ) and image_filename_matches_text(image_url, matching_text)
