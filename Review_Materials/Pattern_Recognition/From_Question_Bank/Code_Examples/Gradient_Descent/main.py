import numpy as np

# 定义训练数据和标签
X = np.array([[1, 2], [2, 2], [1, 1], [2, 0]])
omega = np.array([-1, -1, 1, 1])

# 初始权重和偏置，确保是浮点数类型
w = np.array([-2.0, 2.0])  # 注意这里的浮点数表示
b = -3.0  # 同样确保是浮点数

# 学习率
eta = 0.1

# 梯度下降
for step in range(3):
    # 计算梯度
    grad_w = -np.dot(omega, X)  # 注意这里应该是 omega 和 X 的点积
    grad_b = -np.sum(omega)

    # 更新权重和偏置
    w -= eta * grad_w
    b -= eta * grad_b

# 输出最终的判别函数
print("最终的判别函数 g(x) =", w[0], "* x1 +", w[1], "* x2 +", b)