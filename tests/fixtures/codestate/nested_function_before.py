def outer(values):
    def inner(value):
        return value + 1

    return [inner(value) for value in values]
