#!/usr/bin/env python3
"""Retrieval diagnostics for THRESHER-Chat.

Usage:
    # From CLI
    python -m thresher_chat.debug "how to install thresher"
    python -m thresher_chat.debug "how to install thresher" --k 15 --db-dir /path/to/db
    python -m thresher_chat.debug "how to install thresher" --profile graduate_student
    python -m thresher_chat.debug --list-profiles

    # From Python
    from thresher_chat.debug import diagnose_retrieval
    diagnose_retrieval("how to install thresher")
    diagnose_retrieval("how to install thresher", profile="graduate_student")
"""

import argparse
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from thresher_chat.config import DEFAULT_DB_DIR, DEFAULT_EMBED_MODEL
from thresher_chat.profiles import get_profile_prompt, list_profiles


def diagnose_retrieval(query, db_dir=None, embed_model=None, k=10, profile=None):
    """Run a similarity search and print ranked results with scores.

    Args:
        query: The search query to test.
        db_dir: Path to ChromaDB directory. Defaults to DEFAULT_DB_DIR.
        embed_model: Embedding model name. Defaults to DEFAULT_EMBED_MODEL.
        k: Number of results to return.
        profile: Optional profile key to test augmented query retrieval.

    Returns:
        List of (Document, score) tuples for the original query,
        or both original and augmented if profile is set.
    """
    db_dir = db_dir or DEFAULT_DB_DIR
    embed_model = embed_model or DEFAULT_EMBED_MODEL

    embeddings = OllamaEmbeddings(model=embed_model)
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    # Always run with original query
    print(f"Query:   {query}")
    print(f"DB:      {db_dir}")
    if profile:
        print(f"Profile: {profile}")
    print(f"Top {k} results:")

    results = db.similarity_search_with_score(query, k=k)

    print(f"\n{'=' * 70}")
    print("ORIGINAL QUERY RETRIEVAL")
    print(f"{'=' * 70}")

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        doc_type = doc.metadata.get("type", "?")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n#{i}  score={score:.4f}  [{doc_type}] {source}")
        print(f"    {preview}...")

    # If profile is set, also run with augmented query
    augmented_results = None
    if profile:
        profile_prompt = get_profile_prompt(profile)
        if not profile_prompt:
            print(f"\nWARNING: Profile '{profile}' not found. Available profiles:")
            print(list_profiles())
        else:
            augmented_query = (
                f"[User profile context — tailor your response accordingly:\n"
                f"{profile_prompt}]\n\n"
                f"{query}"
            )
            augmented_results = db.similarity_search_with_score(augmented_query, k=k)

            print(f"\n{'=' * 70}")
            print(f"AUGMENTED QUERY RETRIEVAL (profile: {profile})")
            print(f"{'=' * 70}")

            for i, (doc, score) in enumerate(augmented_results, 1):
                source = doc.metadata.get("source", "?")
                doc_type = doc.metadata.get("type", "?")
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"\n#{i}  score={score:.4f}  [{doc_type}] {source}")
                print(f"    {preview}...")

            # Show ranking changes
            orig_sources = [doc.metadata.get("source", "?") for doc, _ in results]
            aug_sources = [doc.metadata.get("source", "?") for doc, _ in augmented_results]
            if orig_sources != aug_sources:
                print(f"\n{'=' * 70}")
                print("RANKING CHANGES")
                print(f"{'=' * 70}")
                for i, (o, a) in enumerate(zip(orig_sources, aug_sources), 1):
                    marker = " <<" if o != a else ""
                    print(f"  #{i}  {o:40s} -> {a}{marker}")

    print(f"\n{'=' * 70}")
    print(f"Total chunks in DB: {db._collection.count()}")

    return results if not augmented_results else (results, augmented_results)


def main():
    parser = argparse.ArgumentParser(description="THRESHER-Chat retrieval diagnostics")
    parser.add_argument("query", nargs="?", help="Search query to test")
    parser.add_argument("--k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--db-dir", default=None, help="ChromaDB directory")
    parser.add_argument("--embed-model", default=None, help="Embedding model")
    parser.add_argument("--profile", default=None, help="Profile key to test augmented retrieval")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles")
    args = parser.parse_args()

    if args.list_profiles:
        print("Available profiles:")
        print(list_profiles())
        sys.exit(0)

    if not args.query:
        parser.error("query is required (unless using --list-profiles)")

    diagnose_retrieval(args.query, args.db_dir, args.embed_model, args.k, args.profile)


if __name__ == "__main__":
    main()