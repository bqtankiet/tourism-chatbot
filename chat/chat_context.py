class ChatContext:
    def __init__(self):
        self.query = None
        
        # natural language understanding
        self.topic = None
        self.intents = None
        self.ners = {
            "city": None,
            "place": None
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

    def detect_topic(query): pass

    def detect_intent(query): pass
    
    def detect_places(query): pass