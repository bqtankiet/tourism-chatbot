from core.retrieval.topic_detector import TopicDetector
from config import path

TOPIC_DETECTOR_DIR = path.PROJECT_ROOT / "save" / "topic_detector"


class ChatContext:
    def __init__(self, topic_detector_dir=None):
        self.query = None

        # natural language understanding
        self.topic = None
        self.topic_scores = []
        self.intents = []
        self.ners = {
            "city": None,
            "place": None,
        }

        # routing
        self.candidate_places = []
        self.routed_place = None

        # retrieval
        self.retrieved_docs = []
        self.used_index = None

        # memory
        self.summary = []
        self.history = []

        self._topic_detector = None
        self._topic_detector_error = None

        detector_dir = str(topic_detector_dir or TOPIC_DETECTOR_DIR)

        try:
            self._topic_detector = TopicDetector.load(detector_dir)
        except Exception as exc:
            # Keep chat flow alive even if saved detector fails to load.
            self._topic_detector_error = str(exc)

    def reset_turn(self, query):
        self.query = query
        self.topic = None
        self.topic_scores = []
        self.intents = []
        self.ners = {"city": None, "place": None}
        self.candidate_places = []
        self.routed_place = None
        self.retrieved_docs = []
        self.used_index = None

    def detect_topic(self, query=None, top_k=3, threshold=0.3):
        q = (query if query is not None else self.query) or ""

        if not q.strip():
            self.topic = None
            self.topic_scores = []
            return None

        if self._topic_detector is None:
            self.topic = "unknown"
            self.topic_scores = [
                {
                    "topic": "unknown",
                    "score": 0.0,
                    "error": self._topic_detector_error,
                }
            ]
            return self.topic

        results = self._topic_detector.detect(q, top_k=top_k, threshold=threshold)
        self.topic_scores = results
        self.topic = results[0]["topic"] if results else "unknown"
        return self.topic

    def detect_intent(self, query=None):
        q = (query if query is not None else self.query) or ""
        self.intents = ["ask_general"] if q.strip() else []
        return self.intents

    def detect_places(self, query=None):
        _ = (query if query is not None else self.query) or ""
        self.candidate_places = []
        return self.candidate_places

    def update(self, query, top_k=3, threshold=0.3):
        self.reset_turn(query)
        self.detect_topic(top_k=top_k, threshold=threshold)
        self.detect_intent()
        self.detect_places()

        return {
            "query": self.query,
            "topic": self.topic,
            "topic_scores": self.topic_scores,
            "intents": self.intents,
            "ners": self.ners,
        }

    def process_query(self, query):
        return self.update(query)

    def add_turn(self, user_query, assistant_answer, metadata=None):
        self.history.append(
            {
                "user": user_query,
                "assistant": assistant_answer,
                "metadata": metadata or {},
            }
        )

    def to_dict(self):
        return {
            "query": self.query,
            "topic": self.topic,
            "topic_scores": self.topic_scores,
            "intents": self.intents,
            "ners": self.ners,
            "candidate_places": self.candidate_places,
            "routed_place": self.routed_place,
            "retrieved_docs": self.retrieved_docs,
            "used_index": self.used_index,
            "summary": self.summary,
            "history": self.history,
        }