from ast import Raise
from functools import reduce
from itertools import count

from autograd.wrap_util import unary_to_nary

from .tracer import Box, Node, getval, isbox, primitive, toposort, trace
from .util import func, subval

# -------------------- reverse mode --------------------


def make_vjp(fun, x):
    start_node = VJPNode.new_root()
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:

        def vjp(g):
            return vspace(x).zeros()
    else:

        def vjp(g):
            return backward_pass(g, end_node)

    return vjp, end_value


def backward_pass(g, end_node):
    outgrads = {end_node: (g, False)}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        ingrads = node.vjp(outgrad[0])
        for parent, ingrad in zip(node.parents, ingrads):
            outgrads[parent] = add_outgrads(outgrads.get(parent), ingrad)
    return outgrad[0]


class VJPNode(Node):
    __slots__ = ["parents", "vjp"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, "__name__", fun)
            raise NotImplementedError(f"VJP of {fun_name} wrt argnums {parent_argnums} not defined")
        self.vjp = vjpmaker(parent_argnums, value, args, kwargs)

    def initialize_root(self):
        self.parents = []
        self.vjp = lambda g: ()


primitive_vjps = {}


def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker


def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def defvjp(fun, *vjpmakers, **kwargs):
    argnums = kwargs.get("argnums", count())
    vjps_dict = {
        argnum: translate_vjp(vjpmaker, fun, argnum) for argnum, vjpmaker in zip(argnums, vjpmakers)
    }

    def vjp_argnums(argnums, ans, args, kwargs):
        L = len(argnums)
        # These first two cases are just optimizations
        if L == 1:
            argnum = argnums[0]
            try:
                vjpfun = vjps_dict[argnum]
            except KeyError:
                raise NotImplementedError(f"VJP of {fun.__name__} wrt argnum 0 not defined")
            vjp = vjpfun(ans, *args, **kwargs)
            return lambda g: (vjp(g),)
        elif L == 2:
            argnum_0, argnum_1 = argnums
            try:
                vjp_0_fun = vjps_dict[argnum_0]
                vjp_1_fun = vjps_dict[argnum_1]
            except KeyError:
                raise NotImplementedError(f"VJP of {fun.__name__} wrt argnums 0, 1 not defined")
            vjp_0 = vjp_0_fun(ans, *args, **kwargs)
            vjp_1 = vjp_1_fun(ans, *args, **kwargs)
            return lambda g: (vjp_0(g), vjp_1(g))
        else:
            vjps = [vjps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]
            return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def translate_vjp(vjpfun, fun, argnum):
    if vjpfun is None:
        return lambda ans, *args, **kwargs: lambda g: vspace(args[argnum]).zeros()
    elif callable(vjpfun):
        return vjpfun
    else:
        raise Exception(f"Bad VJP '{vjpfun}' for '{fun.__name__}'")


# -------------------- forward mode --------------------


def make_jvp(fun, x):
    def jvp(g):
        start_node = JVPNode.new_root(g)
        end_value, end_node = trace(start_node, fun, x)
        if end_node is None:
            return end_value, vspace(end_value).zeros()
        else:
            return end_value, end_node.g

    return jvp


class JVPNode(Node):
    __slots__ = ["g"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        parent_gs = [parent.g for parent in parents]
        try:
            jvpmaker = primitive_jvps[fun]
        except KeyError:
            name = getattr(fun, "__name__", fun)
            raise NotImplementedError(f"JVP of {name} wrt argnums {parent_argnums} not defined")
        self.g = jvpmaker(parent_argnums, parent_gs, value, args, kwargs)

    def initialize_root(self, g):
        self.g = g


primitive_jvps = {}


def defjvp_argnums(fun, jvpmaker):
    primitive_jvps[fun] = jvpmaker


def defjvp_argnum(fun, jvpmaker):
    def jvp_argnums(argnums, gs, ans, args, kwargs):
        return sum_outgrads(jvpmaker(argnum, g, ans, args, kwargs) for argnum, g in zip(argnums, gs))

    defjvp_argnums(fun, jvp_argnums)


def defjvp(fun, *jvpfuns, **kwargs):
    argnums = kwargs.get("argnums", count())
    jvps_dict = {argnum: translate_jvp(jvpfun, fun, argnum) for argnum, jvpfun in zip(argnums, jvpfuns)}

    def jvp_argnums(argnums, gs, ans, args, kwargs):
        return sum_outgrads(jvps_dict[argnum](g, ans, *args, **kwargs) for argnum, g in zip(argnums, gs))

    defjvp_argnums(fun, jvp_argnums)


def translate_jvp(jvpfun, fun, argnum):
    if jvpfun is None:
        return lambda g, ans, *a, **k: vspace(ans).zeros()
    elif jvpfun == "same":
        return lambda g, ans, *args, **kwargs: fun(*subval(args, argnum, g), **kwargs)
    elif callable(jvpfun):
        return jvpfun
    else:
        raise Exception(f"Bad JVP '{jvpfun}' for '{fun.__name__}'")


def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs: fun(*subval(args, argnum, g), **kwargs))


# -------------------- jac mod ----------------------------
# 为JACNode定义存储雅可比矩阵计算函数的字典
primitive_jacobs = {}

def defjac_argnums(fun, jacmaker):
    primitive_jacobs[fun] = jacmaker


def translate_jac(jacfun, fun, argnum):
    """处理雅可比矩阵计算函数的不同形式"""
    if jacfun is None:
        return lambda ans, *args, **kwargs: vspace(args[argnum]).zeros()
    elif callable(jacfun):
        return jacfun
    else:
        raise Exception(f"Bad Jacobian function '{jacfun}' for '{fun.__name__}'")


def defjac_argnum(fun, jacmaker):
    """为单个参数注册雅可比矩阵计算函数"""
    def jac_argnums(argnums, *args):
        jacs = [jacmaker(argnum, *args) for argnum in argnums]
        return {argnum: jac for argnum, jac in zip(argnums, jacs)}
    
    defjac_argnums(fun, jac_argnums)

def defjac(fun, *jacmakers, **kwargs):
    """为函数的多个参数注册雅可比矩阵计算函数"""
    argnums = kwargs.get("argnums", count())
    jacs_dict = {
        argnum: translate_jac(jacmaker, fun, argnum) 
        for argnum, jacmaker in zip(argnums, jacmakers)
    }

    def jac_argnums(argnums, ans, args, kwargs):
        L = len(argnums)
        # 优化处理常见情况
        if L == 1:
            argnum = argnums[0]
            try:
                jacfun = jacs_dict[argnum]
            except KeyError:
                raise NotImplementedError(f"Jacobian of {fun.__name__} wrt argnum {argnum} not defined")
            jac = jacfun(ans, *args, **kwargs)
            return {argnum: jac}
        elif L == 2:
            argnum_0, argnum_1 = argnums
            try:
                jac_0_fun = jacs_dict[argnum_0]
                jac_1_fun = jacs_dict[argnum_1]
            except KeyError as e:
                missing_argnum = argnum_0 if argnum_0 not in jacs_dict else argnum_1
                raise NotImplementedError(f"Jacobian of {fun.__name__} wrt argnum {missing_argnum} not defined")
            jac_0 = jac_0_fun(ans, *args, **kwargs)
            jac_1 = jac_1_fun(ans, *args, **kwargs)
            return {argnum_0: jac_0, argnum_1: jac_1}
        else:
            jacs_functions = {}
            for argnum in argnums:
                if argnum not in jacs_dict:
                    raise NotImplementedError(f"Jacobian of {fun.__name__} wrt argnum {argnum} not defined")
                jacs_functions[argnum] = jacs_dict[argnum](ans, *args, **kwargs)
            return jacs_functions

    defjac_argnums(fun, jac_argnums)


class JACNode(Node):
    __slots__ = ["parents", "jac_map", "children", "fun"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        self.jac_map = {}
        self.fun = fun
        try:
            jac_maker = primitive_jacobs[fun]
        except KeyError:
            name = getattr(fun, "__name__", fun)
            raise NotImplementedError(f"Jacobian of {name} wrt argnums {parent_argnums} not defined")
        jac_map = jac_maker(parent_argnums, value, args, kwargs)
        for i, parent_node in enumerate(self.parents):
            self.jac_map[parent_node] = jac_map[i]
    def initialize_root(self):
        self.parents = []
        self.jac_map = {}
        self.children = []

    def get_jacobian(self, parent_node):
        """获取当前节点相对于指定父节点的雅可比矩阵"""
        if parent_node in self.jac_map:
            return self.jac_map[parent_node]
        else:
            # 如果未找到特定父节点的雅可比矩阵，返回零矩阵
            for parent in self.parents:
                if parent == parent_node:
                    return vspace(parent._value).zeros()
            raise ValueError(f"Parent node {parent_node} not found")

def transform_graph(end_node):
    """补全计算图中节点与子节点之间的连接
    遍历计算图中的所有节点，为每个节点的父节点添加对当前节点的引用
    """
    # 使用toposort函数按拓扑顺序遍历计算图中的所有节点
    for node in toposort(end_node):
        # 初始化children属性（如果不存在）
        if not hasattr(node, 'children'):
            node.children = []
        
        # 遍历当前节点的所有父节点
        for parent in getattr(node, 'parents', []):
            # 初始化父节点的children属性（如果不存在）
            if not hasattr(parent, 'children'):
                parent.children = []
            
            # 将当前节点添加到父节点的children列表中（如果尚未添加）
            if node not in parent.children:
                parent.children.append(node)

def cross_eliminate(node):
    """跨节点消除雅可比矩阵"""
    if not hasattr(node, 'children'):
        return
    tmp_jac = {}
    for child in node.children:
        for parent in node.parents:
            p2node_jac = node.jac_map[parent]
            node2c_jac = child.jac_map[node]
            # TODO 添加高效的雅可比矩阵乘、推广高阶乘
            new_jac =  node2c_jac @ p2node_jac
            if (parent, child) in tmp_jac:
                tmp_jac[(parent, child)] += new_jac
            else:
                tmp_jac[(parent, child)] = new_jac
    
    for (parent, child), new_jac in tmp_jac.items():
        # 删除当前节点到子节点的连接
        if node in parent.children:
            parent.children.remove(node)
        node.parents = tuple(p for p in node.parents if p is not parent)
        
        child.parents = tuple(p for p in child.parents if p is not node)
        if child in node.children:
            node.children.remove(child)
        # 删除雅可比映射中与当前节点相关的项
        if parent in node.jac_map:
            del node.jac_map[parent]
        if node in child.jac_map:
            del child.jac_map[node]
    for (parent, child), new_jac in tmp_jac.items():
        # 添加新的雅可比矩阵
        if parent in child.jac_map:
            child.jac_map[parent] += new_jac
        else:
            child.jac_map[parent] = new_jac
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents = child.parents + (parent,)


    

def jac_backward_pass(end_node):
    # 首先补全计算图中的连接关系
    transform_graph(end_node)
    topo_res = list(toposort(end_node))
    for node in topo_res[1:-1]:  # 保留首尾节点
        cross_eliminate(node)
    return end_node.jac_map
@unary_to_nary
def make_jac(f, x):
    start_node = JACNode.new_root()
    end_value, end_node = trace(start_node, f, x)
    if end_node is None:
        return {}, end_value
    else:
        transform_graph(end_node)
        jac_map = jac_backward_pass(end_node)[start_node]
        return jac_map, end_value


# -------------------- vector behavior --------------------


def add_outgrads(prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
        vs = vspace(g)
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return sparse_add(vs, prev_g, g), True
            else:
                return vs.mut_add(prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs.mut_add(None, prev_g)
                return sparse_add(vs, prev_g_mutable, g), True
            else:
                return vs.add(prev_g, g), True
    else:
        if sparse:
            return sparse_add(vspace(g), None, g), True
        else:
            return g, False


def sum_outgrads(gs):
    return reduce(add_outgrads, gs, None)[0]


@primitive
def sparse_add(vs, x_prev, x_new):
    x_prev = x_prev if x_prev is not None else vs.zeros()
    return x_new.mut_add(x_prev)


class VSpace:
    __slots__ = []
    mappings = {}
    iscomplex = False

    def __init__(self, value):
        pass

    def zeros(self):
        assert False, repr(self)

    def ones(self):
        assert False, repr(self)

    def standard_basis(self):
        assert False, repr(self)

    def randn(self):
        assert False, repr(self)

    @primitive
    def mut_add(self, x_prev, x_new):
        x_prev = x_prev if x_prev is not None else self.zeros()
        return self._mut_add(x_prev, x_new)

    @primitive
    def add(self, x_prev, x_new):
        return self._add(x_prev, x_new)

    @primitive
    def scalar_mul(self, x, a):
        return self._scalar_mul(x, a)

    @primitive
    def inner_prod(self, x, y):
        return self._inner_prod(x, y)

    @primitive
    def covector(self, x):
        return self._covector(x)

    def _add(self, x, y):
        return x + y

    def _mut_add(self, x, y):
        x += y
        return x

    def _scalar_mul(self, x, a):
        return x * a

    def _inner_prod(self, x, y):
        assert False

    def _covector(self, x):
        return x

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{type(self).__name__}_{self.__dict__}"

    @classmethod
    def register(cls, value_type, vspace_maker=None):
        if vspace_maker:
            VSpace.mappings[value_type] = vspace_maker
        else:
            VSpace.mappings[value_type] = cls


def vspace(value):
    try:
        return VSpace.mappings[type(value)](value)
    except KeyError:
        if isbox(value):
            return vspace(getval(value))
        else:
            raise TypeError(
                "Can't find vector space for value {} of type {}. Valid types are {}".format(
                    value, type(value), VSpace.mappings.keys()
                )
            )


class SparseBox(Box):
    __slots__ = []


class SparseObject:
    __slots__ = ["vs", "mut_add"]

    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add


VSpace.register(SparseObject, lambda x: x.vs)
SparseBox.register(SparseObject)
sparse_object_types = {SparseObject, SparseBox}

# -------------------- core reverse mode grads --------------------

identity_vjp = lambda argnums, *args: lambda g: g
defvjp(sparse_add, None, identity_vjp, identity_vjp)
defvjp(func(VSpace.add), None, identity_vjp, identity_vjp)
defvjp(func(VSpace.mut_add), None, identity_vjp, identity_vjp)
defvjp(
    func(VSpace.inner_prod),
    None,
    lambda ans, vs, x, y: lambda g: vs.covector(vs.scalar_mul(y, g)),
    lambda ans, vs, x, y: lambda g: vs.covector(vs.scalar_mul(x, g)),
)
defvjp(func(VSpace.covector), None, lambda ans, vs, x: lambda g: vs.covector(g))
defvjp(
    func(VSpace.scalar_mul),
    None,
    lambda ans, vs, x, a: lambda g: vs.covector(vs.scalar_mul(vs.covector(g), a)),
    lambda ans, vs, x, a: lambda g: vs.inner_prod(g, vs.covector(x)),
)

# -------------------- core forward mode grads --------------------

identity_jvp = lambda g, *args, **kwargs: g
defjvp(sparse_add, None, identity_jvp, identity_jvp)
defjvp(func(VSpace.mut_add), None, identity_jvp, identity_jvp)
defjvp(func(VSpace.add), None, identity_jvp, identity_jvp)
defjvp(func(VSpace.scalar_mul), None, "same", "same")
defjvp(func(VSpace.inner_prod), None, "same", "same")
defjvp(func(VSpace.covector), None, "same")

# -------------------- deprecation warnings -----------------------

import warnings

deprecated_defvjp_message = """
The {} method is deprecated. See the update guide and tutorial:
https://github.com/HIPS/autograd/blob/master/docs/updateguide.md
https://github.com/HIPS/autograd/blob/master/docs/tutorial.md"""


def deprecated_defvjp(primitive_fun):
    deprecation_msg = deprecated_defvjp_message.format("defvjp")
    vjpfuns = {}

    def defvjp_unstaged(vjpmaker, argnum=0):
        warnings.warn(deprecation_msg)

        def staged_vjpmaker(ans, *args, **kwargs):
            def vjp(g):
                vs, gvs = vspace(args[argnum]), vspace(g)
                return vjpmaker(g, ans, vs, gvs, *args, **kwargs)

            return vjp

        vjpfuns[argnum] = staged_vjpmaker
        argnums, vjpmakers = zip(*[(argnum, vjpfuns[argnum]) for argnum in sorted(vjpfuns.keys())])
        defvjp(primitive_fun, *vjpmakers, argnums=argnums)

    return defvjp_unstaged


def deprecated_defvjp_is_zero(primitive_fun):
    deprecation_msg = deprecated_defvjp_message.format("defvjp_is_zero")
    zero_vjps = [set()]

    def defvjp_is_zero(argnums=(0,)):
        warnings.warn(deprecation_msg)
        zero_vjps[0] |= set(argnums)
        nones = [None] * len(zero_vjps[0])
        defvjp(primitive_fun, *nones, argnums=sorted(zero_vjps[0]))

    return defvjp_is_zero


def deprecated_defgrad(primitive_fun):
    deprecation_msg = deprecated_defvjp_message.format("defgrad")
    gradfuns = {}

    def defgrad(gradfun, argnum=0):
        warnings.warn(deprecation_msg)
        gradfuns[argnum] = gradfun
        argnums, vjpmakers = zip(*[(argnum, gradfuns[argnum]) for argnum in sorted(gradfuns.keys())])
        defvjp(primitive_fun, *vjpmakers, argnums=argnums)

    return defgrad


primitive_ = primitive


def primitive_with_deprecation_warnings(f_raw):
    f_wrapped = primitive_(f_raw)
    f_wrapped.defvjp = deprecated_defvjp(f_wrapped)
    f_wrapped.defvjp_is_zero = deprecated_defvjp_is_zero(f_wrapped)
    f_wrapped.defgrad = deprecated_defgrad(f_wrapped)
    return f_wrapped


primitive = primitive_with_deprecation_warnings
