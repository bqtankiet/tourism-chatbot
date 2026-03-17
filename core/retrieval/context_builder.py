class ContextBuilder:

    def __init__(self, documents: dict):
        self.documents = documents

    def build(self, retrieved_docs, max_docs=5):
        contexts = []

        for doc_id, score in retrieved_docs[:max_docs]:
            if doc_id not in self.documents:
                continue

            content = self.documents[doc_id]

            contexts.append({
                "doc_id": doc_id,
                "score": score,
                "content": content
            })

        return contexts

    def build_text(self, retrieved_docs):
        text = ""
        for (doc_id, score) in retrieved_docs:
            if doc_id not in self.documents:
                continue

            chunk = self.documents[doc_id]

            text += chunk + "\n\n"

        return text