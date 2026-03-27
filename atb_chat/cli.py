#!/usr/bin/env python3
"""ATB-Chat CLI: Entry point with subcommands."""

import argparse
import sys

from atb_chat.config import (
    DEFAULT_CRAWL_DIR,
    DEFAULT_DB_DIR,
    DEFAULT_MODEL,
)

# Subcommand: crawl
def add_crawl_parser(subparsers):
    """Register the 'crawl' subcommand."""
    parser = subparsers.add_parser(
        "crawl",
        help="Crawl ATB sources (preprint, docs, and GitHub) into JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  atb-chat crawl                                    # crawl everything
  atb-chat crawl --sources docs github              # only specific sources
  atb-chat crawl --crawl-dir /path/to/crawl_dir     # custom output directory
""",
    )
    parser.add_argument(
        "-s", "--sources",
        nargs="+",
        choices=["preprint", "docs", "github"],
        default=["preprint", "docs", "github"],
        help="Which sources to crawl (default: all three)",
    )
    parser.add_argument(
        "--crawl-dir",
        default=DEFAULT_CRAWL_DIR,
        dest="crawl-dir",
        help=f"Crawl directory (default: {DEFAULT_CRAWL_DIR})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Debug-level logging",
    )
    parser.set_defaults(func=run_crawl)


def run_crawl(args):
    """Execute the crawl subcommand."""
    from atb_chat.crawler import run
    run(args)


# Subcommand: ingest
def add_ingest_parser(subparsers):
    """Register the 'ingest' subcommand."""
    parser = subparsers.add_parser(
        "ingest",
        help="Index the ATB knowledge base into a ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No default for crawl-dir for now
    parser.add_argument(
        "--crawl-dir", 
        help="Path to the crawl directory created by 'atb-chat crawl'"
        )
    
    parser.add_argument(
        "--db-dir", default=DEFAULT_DB_DIR,
        help=f"Vector DB directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--embed-model", default=None,
        help="Override default embedding model",
    )
    parser.set_defaults(func=run_ingest)


def run_ingest(args):
    """Execute the ingest subcommand."""
    from atb_chat.ingest import run
    run(args)


# Subcommand: chat
def add_chat_parser(subparsers):
    """Register the 'chat' subcommand."""
    parser = subparsers.add_parser(
        "chat",
        help="Start the chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-dir", required=True,
        help="Path to the vector DB directory (created by 'atb-chat ingest')",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override default LLM model",
    )
    parser.add_argument(
        "--embed-model", default=None,
        help="Override default embedding model",
    )
    parser.add_argument(
        "--no-sources", action="store_true",
        help="Don't show source files for each answer",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode (prints retrieved context)"
    )

    parser.set_defaults(func=run_chat)


def run_chat(args):
    """Execute the chat subcommand."""
    from atb_chat.chat import run
    run(args)


# Subcommand: server
def add_server_parser(subparsers):
    """Register the 'server' subcommand."""
    parser = subparsers.add_parser(
        "server",
        help="Start the Flask web UI server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-dir", required=True,
        help="Path to the vector DB directory (created by 'atb-chat ingest')",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override default LLM model",
    )
    parser.add_argument(
        "--embed-model", default=None,
        help="Override default embedding model",
    )
    parser.set_defaults(func=run_server)


def run_server(args):
    """Execute the server subcommand."""
    from atb_chat.server import run
    run(args)


# Main entry point
def main():
    parser = argparse.ArgumentParser(
        prog="atb-chat",
        description="ATB-Chat: Local offline AI assistant for the AllTheBacteria knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    add_crawl_parser(subparsers)
    add_ingest_parser(subparsers)
    add_chat_parser(subparsers)
    add_server_parser(subparsers)
    
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()