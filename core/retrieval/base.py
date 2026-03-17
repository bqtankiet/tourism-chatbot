from abc import ABC, abstractmethod

class BaseRetriever(ABC):

    @abstractmethod
    def search(self, query: str, top_k: int = 5):
        pass

    def search_threshold(self, query: str, threshold: float):
        pass