class Accumulator:
    def reset(self):
        self.total = 100

    def add(self, value):
        self.total = value
        return -1
