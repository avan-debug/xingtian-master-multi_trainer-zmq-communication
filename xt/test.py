
import numpy as np
import matplotlib.pyplot as plt

X=np.array([1, 2, 4, 8, 16]) 
Y=np.array([45, 62, 116, 267, 573])

#定义直线拟合函数
def linear_regression(x, y): 
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
 
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
 
    return np.linalg.solve(A, b)
 
a10, a11 = linear_regression(X, Y)
 
y = 653.9
x = (y - a10) / a11

print("a10 === {}  a11 ========== {} x ========= {}".format(a10, a11, x))
 
# 生成拟合直线的绘制点
_X1 = np.arange (0,20,0.01)
_Y1 = np.array([a10 + a11 * x for x in _X1])
 
#画图
plt.figure(figsize=(6,6))
plt.plot(_X1, _Y1, 'b', linewidth=2) 
plt.legend(bbox_to_anchor=(1,0),loc="lower left")
plt.title("y = {} + {}x".format(a10, a11)) #标题
plt.show()