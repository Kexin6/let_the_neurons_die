import numpy as np
test = {
    2: 1,
    3: 2,
    4: 3,
    5: 0
}

scores = sorted(test, key=test.get)

for elem in scores:
    print(test[elem])

