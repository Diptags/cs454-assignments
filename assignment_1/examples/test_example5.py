import example5


def test_collection_1():
    assert example5.collection(94, 7, -90) == False

def test_collection_2():
    assert example5.collection(42, 24, 3) == False

def test_collection_3():
    assert example5.collection(42, 124, 3) == False

def test_collection_4():
    assert example5.collection(42, 24, 99) == True
