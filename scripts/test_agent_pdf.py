import sys

from agent.graph import run_agent


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python -m "
            "scripts.test_agent_pdf "
            "<pdf_path>"
        )

    pdf_path = sys.argv[1]

    question = (
        "What reinforcement learning "
        "algorithm does EmpRL use?"
    )

    answer = run_agent(
        question,
        pdf_path=pdf_path,
    )

    print()
    print(
        f"Question: {question}"
    )

    print()
    print("Agent answer:")
    print(answer)


if __name__ == "__main__":
    main()