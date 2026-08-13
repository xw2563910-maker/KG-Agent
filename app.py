import argparse

from agent.graph import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "KG-Agent: knowledge-graph-enhanced scientific "
            "research assistant"
        )
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Question to ask KG-Agent. If omitted, interactive input is used.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        help=(
            "Optional local PDF path. Questions mentioning the knowledge "
            "graph together with --pdf use the hybrid KG + PDF route."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("KG-Agent")
    print("-" * 60)

    question = args.question

    if not question:
        question = input("请输入你的问题：").strip()

    try:
        print("\n正在运行 KG-Agent...\n")

        answer = run_agent(
            question,
            pdf_path=args.pdf_path,
        )

        print("KG-Agent 回答：")
        print(answer)

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
    ) as exc:
        print(f"\n错误：{exc}")


if __name__ == "__main__":
    main()
