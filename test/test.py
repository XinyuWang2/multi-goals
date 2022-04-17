from collections import deque
import random


d = deque(maxlen=50)
d.append((1, 2))
d.append((3, 2))
d.append((1, 5))


x = random.sample(d, 2)

print(x)