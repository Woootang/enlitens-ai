"""Command line utilities for embedding ingestion and retrieval testing."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, Optional

from src.agents.context_rag_agent import ContextRAGAgent
from src.models.enlitens_schemas import EnlitensKnowledgeEntry
from src.retrieval.embedding_ingestion import EmbeddingIngestionPipeline, load_knowledge_entries_from_path
from src.retrieval.index_maintenance import IndexMaintenance
from src.retrieval.vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    QdrantVectorStore,
    SearchResult,
)


def build_vector_store(store_type: str, args: argparse.Namespace) -> BaseVectorStore:
    if store_type == "chroma":
        return ChromaVectorStore(
            collection_name=args.collection,
            persist_directory=args.persist_dir,
            embedding_model_name=args.embedding_model,
        )
    return QdrantVectorStore(
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    entries = load_knowledge_entries_from_path(args.knowledge_base)
    vector_store = build_vector_store(args.vector_store, args)
    pipeline = EmbeddingIngestionPipeline(vector_store=vector_store)

    stats = pipeline.ingest_entries(entries, rebuild=args.rebuild)
    total_chunks = sum(stat.chunks_ingested for stat in stats)
    print(f"Ingested {len(stats)} documents ({total_chunks} chunks) into collection '{args.collection}'.")


def cmd_refresh(args: argparse.Namespace) -> None:
    entries = load_knowledge_entries_from_path(args.knowledge_base)
    vector_store = build_vector_store(args.vector_store, args)
    maintenance = IndexMaintenance(EmbeddingIngestionPipeline(vector_store=vector_store))
    report = maintenance.refresh(entries, schedule=args.schedule, rebuild=args.rebuild)
    print(
        f"Refresh complete: {report.documents_processed} documents, "
        f"{report.total_chunks} chunks (schedule={report.schedule})."
    )


def cmd_verify(args: argparse.Namespace) -> None:
    entries = load_knowledge_entries_from_path(args.knowledge_base)
    vector_store = build_vector_store(args.vector_store, args)
    pipeline = EmbeddingIngestionPipeline(vector_store=vector_store)
    report = pipeline.run_integrity_check(entries)

    print(
        f"Integrity report generated at {report.generated_at.isoformat()} - "
        f"expected {report.total_expected}, indexed {report.total_indexed}."
    )
    for item in report.documents:
        status_icon = {"ok": "✅", "missing": "⚠️", "stale": "❌"}.get(item.status, "•")
        print(
            f" {status_icon} {item.document_id}: expected {item.expected_chunks}, "
            f"indexed {item.indexed_chunks} ({item.status})"
        )


def _parse_optional_json(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        print("Failed to parse JSON payload. Provide a valid JSON string.", file=sys.stderr)
        return {}


def _load_entry_by_id(entries: list[EnlitensKnowledgeEntry], document_id: Optional[str]) -> Optional[EnlitensKnowledgeEntry]:
    if not document_id:
        return None
    for entry in entries:
        if entry.metadata.document_id == document_id:
            return entry
    print(f"Document {document_id} not found in provided knowledge base.", file=sys.stderr)
    return None


def _print_search_results(results: list[SearchResult]) -> None:
    if not results:
        print("No results found.")
        return

    for idx, result in enumerate(results, start=1):
        payload = result.payload or {}
        print(f"#{idx} | score={result.score:.4f} | doc={payload.get('document_id')} | type={payload.get('source_type')}")
        print(result.text.strip()[:500])
        print("-")


def cmd_query(args: argparse.Namespace) -> None:
    vector_store = build_vector_store(args.vector_store, args)

    if args.use_agent:
        entries = load_knowledge_entries_from_path(args.knowledge_base) if args.knowledge_base else []
        entry = _load_entry_by_id(entries, args.document_id) if entries else None
        document_text = args.document_text or (entry.full_document_text if entry else "")
        client_insights = _parse_optional_json(args.client_insights)
        founder_insights = _parse_optional_json(args.founder_insights)
        st_louis_context = _parse_optional_json(args.stl_context)

        agent = ContextRAGAgent(vector_store=vector_store, top_k=args.top_k)

        async def run_agent() -> Dict[str, Any]:
            await agent.initialize()
            result = await agent.process(
                {
                    "document_text": document_text or args.prompt,
                    "client_insights": client_insights,
                    "founder_insights": founder_insights,
                    "st_louis_context": st_louis_context,
                    "intermediate_results": {},
                }
            )
            await agent.cleanup()
            return result

        output = asyncio.run(run_agent())
        rag = output.get("rag_retrieval", {})
        print(json.dumps(rag, indent=2, ensure_ascii=False))
        return

    metadata_filter = {"document_id": args.document_id} if args.document_id else None
    results = vector_store.search(args.prompt, limit=args.top_k, metadata_filter=metadata_filter)
    _print_search_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embedding ingestion and retrieval tooling")
    parser.add_argument("--collection", default="enlitens_chunks", help="Vector store collection name")
    parser.add_argument("--vector-store", choices=["qdrant", "chroma"], default="qdrant")
    parser.add_argument("--persist-dir", default=None, help="Chroma persistence directory")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model name")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest knowledge base entries")
    ingest_parser.add_argument("--knowledge-base", required=True)
    ingest_parser.add_argument("--rebuild", action="store_true")
    ingest_parser.set_defaults(func=cmd_ingest)

    refresh_parser = subparsers.add_parser("refresh", help="Run scheduled refresh")
    refresh_parser.add_argument("--knowledge-base", required=True)
    refresh_parser.add_argument("--schedule", choices=["nightly", "weekly"], default="nightly")
    refresh_parser.add_argument("--rebuild", action="store_true")
    refresh_parser.set_defaults(func=cmd_refresh)

    verify_parser = subparsers.add_parser("verify", help="Verify index integrity")
    verify_parser.add_argument("--knowledge-base", required=True)
    verify_parser.set_defaults(func=cmd_verify)

    query_parser = subparsers.add_parser("query", help="Test retrieval for a prompt")
    query_parser.add_argument("prompt", help="Prompt or query text")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--document-id", default=None)
    query_parser.add_argument("--knowledge-base", default=None)
    query_parser.add_argument("--document-text", default=None)
    query_parser.add_argument("--client-insights", default=None, help="JSON string of client themes")
    query_parser.add_argument("--founder-insights", default=None, help="JSON string of founder insights")
    query_parser.add_argument("--stl-context", default=None, help="JSON string of St. Louis context")
    query_parser.add_argument("--use-agent", action="store_true", help="Use the ContextRAG agent for retrieval")
    query_parser.set_defaults(func=cmd_query)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
