class InsufficientImagesError(Exception):
  def __init__(self, expected, received):
    super().__init__(f"Expected {expected} images, but received {received}.")
    self.expected = expected
    self.received = received
