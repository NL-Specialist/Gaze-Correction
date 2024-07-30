import queue

class CircularFrameQueue:
    def __init__(self, maxsize):
        self.queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item):
        if self.queue.full():
            # Remove the oldest item
            self.queue.get()
        self.queue.put(item)

    def get(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def qsize(self):
        return self.queue.qsize()

    def is_empty(self):
        return self.queue.empty()