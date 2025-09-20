from functools import partial

import numpy as onp
from .numpy_boxes import ArrayBox
from . import numpy_wrapper as anp

from autograd.core import make_jac, defjac, primitive

defjac(anp.add, lambda ans, x, y: anp.eye(x.size).reshape(ans.shape + x.shape), lambda ans, x, y: anp.eye(y.size).reshape(ans.shape + y.shape))


defjac(
    anp.multiply,
    lambda ans, x, y: anp.diag(y.flatten()).reshape(ans.shape + x.shape),
    lambda ans, x, y: anp.diag(x.flatten()).reshape(ans.shape + y.shape),
)

defjac(
    anp.divide,
    lambda ans, x, y: anp.diag((1/y).flatten()).reshape(ans.shape + x.shape),
    lambda ans, x, y: anp.diag((-x/y**2).flatten()).reshape(ans.shape + y.shape),
)

