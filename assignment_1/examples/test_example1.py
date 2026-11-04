import example1


def test_foo_1():
    assert example1.foo(94, 7) == -1

def test_foo_2():
    assert example1.foo(42, -9) == 0

def test_foo_3():
    assert example1.foo(42, 0) == 1

def test_bar_1():
    assert example1.bar(-75, 86, -82) == -82

def test_bar_2():
    assert example1.bar(-324, -34, 146) == -34

def test_bar_3():
    assert example1.bar(-24, -94, 772) == -24
