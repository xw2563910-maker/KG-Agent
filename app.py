from llm.client import chat


def main():
    print("KG-Agent")
    print("-" * 40)

    question = input("请输入你的问题：")

    try:
        print("\n正在调用 DeepSeek...\n")

        answer = chat(question)

        print("DeepSeek 回答：")
        print(answer)

    except (ValueError, RuntimeError) as exc:
        print(f"\n错误：{exc}")


if __name__ == "__main__":
    main()