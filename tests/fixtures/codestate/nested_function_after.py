def outer(values):
    def inner(value):
        return value + 2

    return [inner(value) for value in values]
