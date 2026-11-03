"""Command line entry point: ingest, ask, eval, serve."""

from __future__ import annotations

import argparse

from .config import settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="citedrag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="chunk a folder of .md/.txt/.pdf files into an index")
    p.add_argument("folder")

    p = sub.add_parser("ask", help="ask a question and get a cited answer")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=settings.top_k)

    p = sub.add_parser("eval", help="run the golden question set")
    p.add_argument("--golden", default="eval/golden.jsonl")
    p.add_argument("-k", type=int, default=settings.top_k)
    p.add_argument("--judge", action="store_true", help="also generate answers and score them with the LLM judge")

    p = sub.add_parser("serve", help="run the FastAPI server")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    from .ingest import ingest_folder, load_index, save_index
    from .retrieve import Retriever

    if args.cmd == "ingest":
        chunks = ingest_folder(args.folder, settings.chunk_words, settings.chunk_overlap)
        save_index(chunks, settings.index_path)
        print(f"Indexed {len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents -> {settings.index_path}")

    elif args.cmd == "ask":
        from .generate import respond
        retriever = Retriever(load_index(settings.index_path))
        result = respond(args.question, retriever.search(args.question, k=args.k))
        if result.mode == "extractive":
            print("(extractive mode: set ANTHROPIC_API_KEY for AI-written answers)\n")
        print(result.text, "\n")
        for c in result.citations:
            print(f"[{c.number}] {c.label}: \"{c.quote}\"")

    elif args.cmd == "eval":
        from .evaluate import load_golden, run_eval, write_report
        retriever = Retriever(load_index(settings.index_path))
        result = run_eval(retriever, load_golden(args.golden), k=args.k, use_llm=args.judge)
        for key, value in result["summary"].items():
            print(f"{key:>20}: {value}")
        print(f"\nReport: {write_report(result)}")

    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("citedrag.api:app", port=args.port)


if __name__ == "__main__":
    main()
