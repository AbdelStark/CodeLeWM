class Accumulator:
    def reset(self):
        self.total = 0

    def add(self, value):
        self.total = self.total + value
        return self.total
