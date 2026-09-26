import example3


def test_intersect_1():
    assert example3.intersect(94, 7, -90, -34, 30, 24, 3, 100) == False

def test_intersect_2():
    assert example3.intersect(22, -9, 49, -45, 29, -65, -28, -65) == False

def test_intersect_3():
    assert example3.intersect(869, -582, 979, 131, -24, -94, 772, 67) == False

def test_intersect_4():
    assert example3.intersect(-85, 40, -97, -77, 84, 2, 81, 100) == False

def test_intersect_5():
    assert example3.intersect(-6, 7, 4, -8, -8, 0, 6, 5) == True

def test_intersect_6():
    assert example3.intersect(-1, 7, -1, -7, 7, 0, 7, -4) == False

def test_intersect_7():
    assert example3.intersect(-1, 7, -1, -7, 7, -4, 7, -4) == True

def test_intersect_8():
    assert example3.intersect(-1, -7, -1, -7, 7, 0, 7, -4) == True
