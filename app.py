from agent.graph import run_agent


def main():
    print("KG-Agent")
    print("-" * 40)

    question = input("请输入你的问题：")

    try:
        print("\n正在运行 KG-Agent...\n")

        answer = run_agent(question)

        print("KG-Agent 回答：")
        print(answer)

    except (ValueError, RuntimeError) as exc:
        print(f"\n错误：{exc}")


if __name__ == "__main__":
    main()