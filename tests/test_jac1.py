from autograd.core import make_jac
import numpy as np
def foo(x, y, z):
    return (x * y) + y + z

x = np.random.rand(3)
y = np.random.rand(3)
z = np.random.rand(3)

jac, val = make_jac(foo, argnum=[0,1,2])(x, y, z) # jac shape is [3 output, 9 input]
print(jac, val)

def check_jac(jac, val, x, y, z):
    for i in range(3):
        assert np.isclose(jac[i, i], y[i])
        assert np.isclose(jac[i, i+3], x[i] + 1)
        assert np.isclose(jac[i, i+6], 1)   


check_jac(jac, val, x, y, z)
