import numpy as np
from autograd.core import make_jac, JACNode

# 定义函数
def foo(x, y, z):
    return (x * y) + y + z

# 创建输入
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
z = np.array([7.0, 8.0, 9.0])

# 直接使用make_jac函数
jac_map, val = make_jac(foo, argnum=[0, 1, 2])(x, y, z)

print("雅可比矩阵映射:")
for node, jac in jac_map.items():
    print(f"节点 {node}:")
    print(jac)
    print()

print("计算结果:", val)