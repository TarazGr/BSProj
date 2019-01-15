def accuracy(y_true, y_pred):
    assert(len(y_true) == len(y_pred))
    correct = 0
    for y1, y2 in zip(y_true, y_pred):
        if y1 == y2:
            correct += 1
    return correct / len(y_true)
