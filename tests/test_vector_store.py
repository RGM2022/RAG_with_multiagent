import sys
import types
import unittest
from email.message import Message
from io import BytesIO
from math import sqrt
from pathlib import Path
import socket
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.search_engine import (
    ImageAsset,
    SearchResult,
    SourceDocument,
    chunk_documents,
    fetch_image,
    tokenize,
)
from vector_store import (
    IMAGE_MIN_SIMILARITY,
    VECTOR_CHUNK_OVERLAP,
    VECTOR_CHUNK_SIZE,
    ChromaVectorStore,
    VectorSearchResult,
    image_filename_matches_query,
    image_filename_matches_text,
)


class _FakeCollection:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def get(self, where=None, include=None):
        records = self.records
        if where:
            key, value = next(iter(where.items()))
            allowed = value.get("$in", []) if isinstance(value, dict) else [value]
            records = {
                identifier: record
                for identifier, record in records.items()
                if record["metadata"].get(key) in allowed
            }
        return {
            "ids": list(records),
            "documents": [record["document"] for record in records.values()],
            "metadatas": [record["metadata"] for record in records.values()],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for identifier, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.records[identifier] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def delete(self, ids):
        for identifier in ids:
            self.records.pop(identifier, None)

    def query(self, query_embeddings, n_results, include, where=None):
        query = query_embeddings[0]
        records = self.records
        if where:
            def matches_filter(metadata, filter_expression):
                if "$or" in filter_expression:
                    return any(
                        matches_filter(metadata, condition)
                        for condition in filter_expression["$or"]
                    )
                key, value = next(iter(filter_expression.items()))
                allowed = value.get("$in", []) if isinstance(value, dict) else [value]
                return metadata.get(key) in allowed

            records = {
                identifier: record
                for identifier, record in records.items()
                if matches_filter(record["metadata"], where)
            }

        def cosine_distance(vector):
            dot = sum(left * right for left, right in zip(vector, query))
            vector_norm = sqrt(sum(value * value for value in vector))
            query_norm = sqrt(sum(value * value for value in query))
            return 1.0 - dot / (vector_norm * query_norm)

        ranked = sorted(
            records.items(),
            key=lambda item: cosine_distance(item[1]["embedding"]),
        )[:n_results]
        return {
            "ids": [[identifier for identifier, _ in ranked]],
            "documents": [[record["document"] for _, record in ranked]],
            "metadatas": [[record["metadata"] for _, record in ranked]],
            "distances": [
                [
                    cosine_distance(record["embedding"])
                    for _, record in ranked
                ]
            ],
        }


class _FakeClient:
    def __init__(self, path):
        self.collections = {}

    def get_or_create_collection(self, name, metadata):
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeEmbedder:
    def encode(self, sentences, *, normalize_embeddings):
        return [[float(len(sentence)), 1.0] for sentence in sentences]


class _FakeImageTextEmbedder:
    def encode(self, inputs, *, normalize_embeddings):
        return [[1.0, 0.0] for _ in inputs]


class _ScoredImageTextEmbedder:
    def encode(self, inputs, *, normalize_embeddings):
        if inputs and isinstance(inputs[0], str):
            similarity = 0.274
            return [[similarity, sqrt(1.0 - similarity**2)]]
        return [[1.0, 0.0] for _ in inputs]


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient("unused")
        chromadb = types.SimpleNamespace(PersistentClient=lambda path: self.client)
        self.chromadb_patch = patch.dict(sys.modules, {"chromadb": chromadb})
        self.chromadb_patch.start()
        self.addCleanup(self.chromadb_patch.stop)
        self.store = ChromaVectorStore(Path("unused"))
        self.embedder = _FakeEmbedder()

    def test_indexing_same_source_replaces_stale_passages(self):
        original = SearchResult("guide.txt", None, 1, "old passage", 0.0)
        updated = SearchResult("guide.txt", None, 1, "new passage", 0.0)

        self.store.index_source("upload:guide.txt", [original], self.embedder)
        self.store.index_source("upload:guide.txt", [updated], self.embedder)

        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.list_sources(), ["guide.txt"])
        result = self.store.search("new passage", self.embedder)
        self.assertEqual(result[0].text, "new passage")
        self.assertEqual(result[0].citation, "guide.txt, passage 1")

    def test_indexing_preserves_page_and_source_url_citations(self):
        passage = SearchResult(
            "guide.pdf",
            3,
            2,
            "network troubleshooting",
            0.0,
            source_url="https://example.com/guide#network",
        )

        self.store.index_source("knowledge_base:guide.pdf", [passage], self.embedder)
        result = self.store.search("network troubleshooting", self.embedder)[0]

        self.assertEqual(result.citation, "guide.pdf, page 3, passage 2")
        self.assertEqual(result.source_url, "https://example.com/guide#network")

    def test_search_hides_passages_below_minimum_similarity(self):
        passage = SearchResult("guide.txt", None, 1, "an unrelated topic", 0.0)
        self.store.index_source("upload:guide.txt", [passage], self.embedder)

        results = self.store.search(
            "a completely different query",
            self.embedder,
            min_similarity=1.0,
        )

        self.assertEqual(results, [])
        fallback_results = self.store.search(
            "a completely different query",
            self.embedder,
            min_similarity=-1.0,
        )
        self.assertEqual([result.text for result in fallback_results], ["an unrelated topic"])

    def test_search_can_be_limited_to_selected_sources(self):
        passages = [
            SearchResult("guide.txt", None, 1, "network setup", 0.0),
            SearchResult("notes.txt", None, 1, "network setup", 0.0),
        ]
        self.store.index_source("upload:guide.txt", [passages[0]], self.embedder)
        self.store.index_source("upload:notes.txt", [passages[1]], self.embedder)

        results = self.store.search(
            "network setup",
            self.embedder,
            sources=["notes.txt"],
        )

        self.assertEqual([result.source for result in results], ["notes.txt"])

    def test_search_matches_webpage_url_when_text_source_is_page_title(self):
        page_url = "https://example.com/guide"
        passage = SearchResult(
            "Example Guide",
            None,
            1,
            "Types of memory include sensory, short-term, and long-term memory.",
            0.0,
            source_url=page_url,
        )
        self.store.index_source("website:guide", [passage], self.embedder)

        results = self.store.search(
            "types of memory",
            self.embedder,
            sources=[page_url],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "Example Guide")

    def test_adjacent_relevant_passages_are_merged_into_complete_context(self):
        page_url = "https://example.com/agents#hugging-gpt"
        passages = [
            SearchResult("Agents", None, 27, "(1) Task planning details.", 0.8, source_url=page_url),
            SearchResult("Agents", None, 28, "(2) Model selection details.", 0.4, source_url=page_url),
            SearchResult("Agents", None, 29, "(3) Task execution details.", 0.4, source_url=page_url),
            SearchResult("Agents", None, 30, "(4) Response generation details.", 0.7, source_url=page_url),
        ]
        self.store.index_source("website:agents", passages, self.embedder)

        merged = self.store.merge_adjacent_passages(
            [
                VectorSearchResult(
                    passage.source,
                    passage.page,
                    passage.chunk_id,
                    passage.text,
                    passage.score,
                    passage.source_url,
                )
                for passage in (passages[0], passages[3])
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertIn("(1) Task planning", merged[0].text)
        self.assertIn("(2) Model selection", merged[0].text)
        self.assertIn("(3) Task execution", merged[0].text)
        self.assertIn("(4) Response generation", merged[0].text)
        self.assertAlmostEqual(merged[0].similarity, 0.8)

    def test_single_webpage_match_includes_remaining_chunks_from_its_section(self):
        page_url = "https://example.com/agents#challenges"
        passages = [
            SearchResult("Agents", None, 60, "Finite context and planning challenges.", 0.8, source_url=page_url),
            SearchResult("Agents", None, 61, "Reliability of natural language interface.", 0.3, source_url=page_url),
            SearchResult(
                "Agents",
                None,
                62,
                "Case studies describe separate examples.",
                0.2,
                source_url="https://example.com/agents#case-studies",
            ),
        ]
        self.store.index_source("website:agents", passages, self.embedder)

        merged = self.store.merge_adjacent_passages(
            [
                VectorSearchResult(
                    passages[0].source,
                    passages[0].page,
                    passages[0].chunk_id,
                    passages[0].text,
                    passages[0].score,
                    passages[0].source_url,
                )
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertIn("Finite context", merged[0].text)
        self.assertIn("Reliability of natural language interface", merged[0].text)
        self.assertNotIn("Case studies", merged[0].text)

    def test_heading_only_result_expands_into_next_webpage_passages(self):
        base_url = "https://example.com/guide"
        passages = [
            SearchResult("Guide", None, 35, "Case Studies", 0.9, source_url=f"{base_url}#case-studies"),
            SearchResult(
                "Guide",
                None,
                36,
                "Scientific Discovery Agent. ChemCrow uses expert tools for chemistry tasks.",
                0.5,
                source_url=f"{base_url}#scientific-discovery-agent",
            ),
            SearchResult(
                "Guide",
                None,
                37,
                "The workflow combines reasoning with domain-specific tools.",
                0.4,
                source_url=f"{base_url}#scientific-discovery-agent",
            ),
            SearchResult(
                "Guide",
                None,
                38,
                "Generative Agents Simulation. Twenty-five virtual characters interact.",
                0.3,
                source_url=f"{base_url}#generative-agents-simulation",
            ),
        ]
        self.store.index_source("website:guide", passages, self.embedder)
        heading_result = VectorSearchResult(
            "Guide",
            None,
            35,
            "Case Studies",
            0.9,
            f"{base_url}#case-studies",
        )

        expanded = self.store.merge_adjacent_passages([heading_result])

        self.assertEqual(len(expanded), 1)
        self.assertIn("Case Studies", expanded[0].text)
        self.assertIn("Scientific Discovery Agent", expanded[0].text)
        self.assertIn("combines reasoning", expanded[0].text)
        self.assertIn("Generative Agents Simulation", expanded[0].text)

    def test_image_filename_can_match_retrieved_section_text(self):
        self.assertTrue(
            image_filename_matches_text(
                "https://example.com/hugging-gpt.png",
                "Illustration of how HuggingGPT works.",
            )
        )
        self.assertFalse(
            image_filename_matches_text(
                "https://example.com/mips.png",
                "Illustration of how HuggingGPT works.",
            )
        )

    def test_low_similarity_named_image_must_match_query_and_retrieved_text(self):
        challenge_text = (
            "Challenges include reliability of the natural language interface "
            "between the model, memory, and external tools."
        )
        self.assertFalse(
            image_filename_matches_query(
                "https://example.com/memory.png",
                "explain Challenges",
                challenge_text,
            )
        )
        self.assertTrue(
            image_filename_matches_query(
                "https://example.com/hugging-gpt.png",
                "explain HuggingGPT",
                "Illustration of how HuggingGPT works.",
            )
        )

    def test_image_filename_acronym_matches_its_expanded_query(self):
        self.assertTrue(
            image_filename_matches_query(
                "https://example.com/CoH.png",
                "what is Chain of Hindsight",
                "Chain of Hindsight (CoH) encourages the model to improve.",
            )
        )

    def test_search_rejects_out_of_range_similarity_threshold(self):
        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            self.store.search("query", self.embedder, min_similarity=1.1)

    def test_embedding_chunks_stay_within_the_configured_token_limit(self):
        document = SourceDocument(
            "guide.txt",
            " ".join(f"word{index}" for index in range(VECTOR_CHUNK_SIZE * 3)),
        )

        passages = chunk_documents(
            [document],
            chunk_size=VECTOR_CHUNK_SIZE,
            overlap=VECTOR_CHUNK_OVERLAP,
        )

        self.assertGreater(len(passages), 1)
        self.assertTrue(
            all(len(tokenize(passage.text)) <= VECTOR_CHUNK_SIZE for passage in passages)
        )

    def test_image_indexing_replaces_stale_images_and_searches_by_text(self):
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), color="red").save(image_bytes, format="PNG")
        first_image = ImageAsset(
            "https://example.com/old.png",
            "https://example.com/guide",
            "Old network diagram",
            image_bytes.getvalue(),
        )
        updated_image = ImageAsset(
            "https://example.com/new.png",
            "https://example.com/guide",
            "Updated network diagram",
            image_bytes.getvalue(),
        )
        embedder = _FakeImageTextEmbedder()

        self.store.index_images("website-images:guide", [first_image], embedder)
        self.store.index_images("website-images:guide", [updated_image], embedder)

        self.assertEqual(self.store.image_count(), 1)
        self.assertIn("https://example.com/guide", self.store.list_sources())
        results = self.store.search_images(
            "network diagram",
            embedder,
            sources=["https://example.com/guide"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].image_url, "https://example.com/new.png")
        self.assertEqual(results[0].page_url, "https://example.com/guide")
        self.assertEqual(results[0].alt_text, "Updated network diagram")

    def test_image_search_accepts_relevant_result_at_calibrated_image_threshold(self):
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), color="red").save(image_bytes, format="PNG")
        image = ImageAsset(
            "https://example.com/mips.png",
            "https://example.com/guide",
            "MIPS diagram",
            image_bytes.getvalue(),
        )
        embedder = _ScoredImageTextEmbedder()
        self.store.index_images("website-images:guide", [image], embedder)

        results = self.store.search_images(
            "Maximum Inner Product Search MIPS",
            embedder,
            min_similarity=IMAGE_MIN_SIMILARITY,
            sources=["https://example.com/guide"],
        )
        stricter_results = self.store.search_images(
            "Maximum Inner Product Search MIPS",
            embedder,
            min_similarity=0.275,
            sources=["https://example.com/guide"],
        )

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].similarity, 0.274)
        self.assertEqual(stricter_results, [])

    def test_fetch_image_validates_and_returns_image_content(self):
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), color="blue").save(image_bytes, format="PNG")
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = Message()
        response.headers["Content-Type"] = "image/png"
        response.read.return_value = image_bytes.getvalue()
        response.geturl.return_value = "https://cdn.example.com/diagram.png"
        public_dns_result = (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 443),
        )
        with (
            patch("RAG.search_engine.socket.getaddrinfo", return_value=[public_dns_result]),
            patch("RAG.search_engine.build_opener") as build_opener,
        ):
            build_opener.return_value.open.return_value = response
            asset = fetch_image(
                "https://example.com/diagram.png",
                page_url="https://example.com/guide",
                alt_text="Network diagram",
            )

        self.assertEqual(asset.image_url, "https://cdn.example.com/diagram.png")
        self.assertEqual(asset.page_url, "https://example.com/guide")
        self.assertEqual(asset.alt_text, "Network diagram")
        self.assertEqual(asset.content, image_bytes.getvalue())


if __name__ == "__main__":
    unittest.main()
