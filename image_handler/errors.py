class InsufficientImagesError(Exception):
    def __init__(self, expected, received):
        super().__init__(f"Expected {expected} images, but received {received}.")
        self.expected = expected
        self.received = received


class TooManyRequestsError(Exception):
    def __init__(self):
        super().__init__("Exceeded maximum number of requests.")


class OllamaError(Exception):
    def __init__(self):
        super().__init__("Ensure Ollama is running locally.")
