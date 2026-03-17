ROLE = "Bạn là chatbot tư vấn du lịch tại Việt Nam. Luôn trả lời bằng tiếng việt."

POLICY = """
- Sử dụng thông tin trong [DỮ LIỆU] để trả lời
- Không dùng các kiến thức không liên quan đến ngành du lịch
- Nếu không có thông tin → nói: "Mình không có thông tin về câu hỏi này."
- Nếu user trò chuyện bình thường như xin chào, hỏi vai trò, cảm ơn, hoặc tương tự thì được phép trả lời lại.
- Nếu không liên quan du lịch thì từ chối lịch sự theo phong cách: Xin lỗi mình không thể trả lời câu hỏi của bạn ngay lúc này. Nếu bạn có câu hỏi khác về du lịch hoặc cần tư vấn, hãy cho mình biết nhé!
- Nếu [DỮ LIỆU] rỗng thì không được tự suy diễn dữ kiện du lịch
"""

STYLE = """
- Thân thiện, tự nhiên như hướng dẫn viên
- Ngắn gọn, dễ hiểu
- Có thể dùng emoji vui nhộn 🙂
"""

FORMAT = """
- Trả lời rõ ràng
- Ưu tiên bullet points nếu phù hợp
- Luôn trả lời bằng tiếng việt
"""


class PromptBuilder:

    def __init__(self):
        self.role = ROLE
        self.policy = POLICY
        self.style = STYLE
        self.format = FORMAT

    def build_system_prompt(self):
        return (
            f"[VAI TRÒ]\n{self.role}\n\n"
            f"[QUY TẮC]\n{self.policy}\n\n"
            f"[ĐỊNH DẠNG TRẢ LỜI]\n{self.format}\n\n"
            f"[PHONG CÁCH]\n{self.style}"
        )

    def build_user_prompt(self, query, context):
        safe_context = context.strip()
        if not safe_context:
            safe_context = "(Không có dữ liệu truy hồi phù hợp)"

        return (
            "Tôi cần hỗ trợ về vấn đề du lịch" # Để topic luôn là du lịch trong mỗi request
            f"[DỮ LIỆU]\n{safe_context}\n\n"
            f"[CÂU HỎI]\n{query}\n\n"
            "[TRẢ LỜI]"
        )

    def build_messages(self, query, context):
        return [
            {
                "role": "system",
                "content": self.build_system_prompt(),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(query, context),
            },
        ]

    def build(self, query, context):
        messages = self.build_messages(query, context)
        return "\n\n".join([m["content"] for m in messages])