import re

def should_skip_retrieval(query: str) -> bool:
    q = query.strip().lower()

    if not q:
        return True

    # Các truy vấn xã giao/điều khiển không cần retrieval.
    patterns = [
        r"^(hi|hello|hey|helo|heloo|xin chao|xin chào|chao|chào|helo)\b",
        r"\b(chao buoi sang|chào buổi sáng|chao buoi toi|chào buổi tối|good morning|good evening)\b",
        r"\b(cam on|cảm ơn|thanks|thank you|thx|tks|ty)\b",
        r"\b(ban la ai|bạn là ai|ban ten gi|bạn tên gì|ten ban la gi|tên bạn là gì)\b",
        r"\b(vai tro cua ban|vai trò của bạn|ban lam duoc gi|bạn làm được gì|you are who|what can you do)\b",
        r"^(ok|oke|okay|okie|okela|yes|yeah|yep|uhm|ừ|ừm|um)\b",
        r"\b(duoc|được|dong y|đồng ý|xac nhan|xác nhận)\b",
        r"\b(khong|không|no|nope|nah|chua can|chưa cần|thoi)\b",
        r"^(bye|goodbye|tam biet|tạm biệt|bye bye|see you|good|goodjob|good job|wow|perfect)\b",
        r"\b(hẹn gặp lại|hen gap lai|see ya|later)\b",
        r"\b(reset|clear|xoa|xóa|lam moi|làm mới|start over|restart)\b",
        r"^(bot|ai|chatbot|assistant)\b",
        r"\b(khong biet|không biết|idk|i dont know|whatever)\b",
        r"\b(gi cung duoc|gì cũng được|anything|anyway)\b",
        r"\b(doi ti|đợi tí|wait a sec|hold on|khoan)\b",
        r"\b(hom nay the nao|hôm nay thế nào|how are you|how r u|ban khoe khong|bạn khỏe không)\b",
    ]

    return any(re.fullmatch(p, q) for p in patterns)