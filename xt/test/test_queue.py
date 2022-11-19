from multiprocessing import Queue
from time import time
import numpy as np


if __name__ == "__main__":
    queue = Queue()
    a = np.zeros(shape=[1, 84, 84, 4], dtype=np.uint8)
    queue.put(a)
    start_0 = time()
    p = queue.get()
    end_0 = time()
    print("cost time ============= {}".format(end_0 - start_0))
    # print(p)

