from chat.chat_engine import ChatEngine

chat_engine = ChatEngine()

if __name__ == "__main__":
    print("🚀 Chat system ready!")

    while True:
        q = input("\n💬 Nhập câu hỏi (exit để thoát): ")

        if q.lower() == "exit":
            break

        res = chat_engine.chat(q)

        print("\n⚙️ DEBUG:")
        print(res.debug)

        print("\n🤖 Trả lời:\n")
        print(res.answer)

        print("---" * 10)