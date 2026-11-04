def testme(a: int, b: int, c: int) -> None:
	while a < b:
		if c > 57 and c < 284:
			a += 1
		elif b - c > 2048 or a - c < 256 :
			a -= 1

		if a < 0:
			break
	return