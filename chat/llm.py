from litellm import completion
from chat.prompt import PromptBuilder


class LLM:

    def __init__(
        self,
        model="ollama/qwen3:1.7b",
        temperature=0.3,
        max_tokens=1024
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.prompt_builder = PromptBuilder()

    def generate(self, query, context, stream=False):
        messages = self.prompt_builder.build_messages(query, context)

        if stream:
            return self._stream(messages)

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response["choices"][0]["message"]["content"]

    def _stream(self, messages):
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        for chunk in response:
            delta = chunk["choices"][0]["delta"]

            if "content" in delta and delta["content"] is not None:
                yield delta["content"]