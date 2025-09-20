from autograd.core import make_jac
import numpy as np

# 添加调试信息
import autograd.core as core
from autograd.tracer import toposort

# 保存原始的jac_backward_pass函数
original_jac_backward_pass = core.jac_backward_pass

# 保存原始的cross_eliminate函数
original_cross_eliminate = core.cross_eliminate

# 定义带调试信息的jac_backward_pass函数
def debug_jac_backward_pass(end_node):
    print("--- Topological sort order ---")
    # 首先补全计算图中的连接关系
    core.transform_graph(end_node)
    
    # 显示拓扑排序的结果
    sorted_nodes = list(toposort(end_node))
    for i, node in enumerate(sorted_nodes):
        if hasattr(node, 'fun'):
            print(f"  {i}: {node.fun.__name__}")
        else:
            print(f"  {i}: {node}")
    
    print("--- Processing nodes in topological order ---")
    for node in sorted_nodes:
        if hasattr(node, 'fun'):
            print('eliminating node', node.fun.__name__)
        else:
            print('eliminating node', node)
        core.cross_eliminate(node)
    return end_node.jac_map

# 定义带调试信息的cross_eliminate函数
def debug_cross_eliminate(node):
    print(f"--- Processing node {node.fun.__name__ if hasattr(node, 'fun') else str(node)} ---")
    print(f"  Has children attribute: {hasattr(node, 'children')}")
    if hasattr(node, 'children'):
        print(f"  Children count: {len(node.children)}")
        for i, child in enumerate(node.children):
            print(f"    Child {i}: {child.fun.__name__ if hasattr(child, 'fun') else str(child)}")
    print(f"  Has parents attribute: {hasattr(node, 'parents')}")
    if hasattr(node, 'parents'):
        print(f"  Parents count: {len(node.parents)}")
        for i, parent in enumerate(node.parents):
            print(f"    Parent {i}: {parent.fun.__name__ if hasattr(parent, 'fun') else str(parent)}")
    print(f"  Has jac_map attribute: {hasattr(node, 'jac_map')}")
    if hasattr(node, 'jac_map'):
        print(f"  Jacobian map entries: {len(node.jac_map)}")
        for parent, jac in node.jac_map.items():
            if hasattr(parent, 'fun'):
                print(f"    To parent {parent.fun.__name__}:")
            else:
                print(f"    To parent {parent}:")
            print(f"      {jac}")
    
    # 调用原始函数
    try:
        return original_cross_eliminate(node)
    except Exception as e:
        print(f"  Error in cross_eliminate: {e}")
        # 打印更多调试信息
        if hasattr(node, 'children'):
            for child in node.children:
                print(f"    Child {child.fun.__name__ if hasattr(child, 'fun') else str(child)} jac_map keys:")
                for key in child.jac_map.keys():
                    if hasattr(key, 'fun'):
                        print(f"      {key.fun.__name__}")
                    else:
                        print(f"      {key}")
        return None

# 替换原始函数
core.jac_backward_pass = debug_jac_backward_pass
core.cross_eliminate = debug_cross_eliminate

def foo(x, y, z):
    return (x * y) + y + z

def print_computation_graph(end_node):
    """打印计算图结构的函数"""
    print("\n=== 计算图结构 ===")
    
    # 首先补全计算图中的连接关系
    core.transform_graph(end_node)
    
    # 获取所有节点
    all_nodes = list(toposort(end_node))
    
    # 创建节点ID映射
    node_ids = {}
    for i, node in enumerate(all_nodes):
        node_ids[node] = i
    
    # 打印节点信息
    print("\n节点列表:")
    for i, node in enumerate(all_nodes):
        if hasattr(node, 'fun'):
            print(f"  节点 {i}: {node.fun.__name__}")
        else:
            print(f"  节点 {i}: {type(node).__name__}")
    
    # 打印边信息
    print("\n边连接关系:")
    for i, node in enumerate(all_nodes):
        if hasattr(node, 'parents'):
            for parent in node.parents:
                if parent in node_ids:
                    print(f"  节点 {node_ids[parent]} -> 节点 {i}")
        
        if hasattr(node, 'children'):
            for child in node.children:
                if child in node_ids:
                    print(f"  节点 {i} -> 节点 {node_ids[child]}")
    
    # 打印雅可比矩阵映射
    print("\n雅可比矩阵映射:")
    for i, node in enumerate(all_nodes):
        if hasattr(node, 'jac_map') and node.jac_map:
            node_name = node.fun.__name__ if hasattr(node, 'fun') else str(node)
            print(f"  节点 {i} ({node_name}) 的雅可比映射:")
            for parent, jac in node.jac_map.items():
                if parent in node_ids:
                    parent_name = parent.fun.__name__ if hasattr(parent, 'fun') else str(parent)
                    print(f"    来自节点 {node_ids[parent]} ({parent_name}) 的雅可比矩阵:")
                    print(f"      {jac}")
    
    print("\n=== 计算图结构结束 ===\n")

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
z = np.array([7.0, 8.0, 9.0])

# 使用make_jac函数创建计算图并打印计算图结构
def print_graph_from_jac():
    # 首先保存原始的jac_backward_pass函数
    original_jac_backward_pass = core.jac_backward_pass
    
    # 定义一个只打印计算图结构但不计算雅可比矩阵的函数
    def print_only_jac_backward_pass(end_node):
        print("\n=== 计算图结构 ===")
        
        # 首先补全计算图中的连接关系
        core.transform_graph(end_node)
        
        # 获取所有节点
        all_nodes = list(toposort(end_node))
        
        # 创建节点ID映射
        node_ids = {}
        for i, node in enumerate(all_nodes):
            node_ids[node] = i
        
        # 打印节点信息
        print("\n节点列表:")
        for i, node in enumerate(all_nodes):
            if hasattr(node, 'fun'):
                print(f"  节点 {i}: {node.fun.__name__}")
            else:
                print(f"  节点 {i}: {type(node).__name__}")
        
        # 打印边信息
        print("\n边连接关系:")
        for i, node in enumerate(all_nodes):
            if hasattr(node, 'parents'):
                for parent in node.parents:
                    if parent in node_ids:
                        print(f"  节点 {node_ids[parent]} -> 节点 {i}")
            
            if hasattr(node, 'children'):
                for child in node.children:
                    if child in node_ids:
                        print(f"  节点 {i} -> 节点 {node_ids[child]}")
        
        # 打印雅可比矩阵映射
        print("\n雅可比矩阵映射:")
        for i, node in enumerate(all_nodes):
            if hasattr(node, 'jac_map') and node.jac_map:
                node_name = node.fun.__name__ if hasattr(node, 'fun') else str(node)
                print(f"  节点 {i} ({node_name}) 的雅可比映射:")
                for parent, jac in node.jac_map.items():
                    if parent in node_ids:
                        parent_name = parent.fun.__name__ if hasattr(parent, 'fun') else str(parent)
                        print(f"    来自节点 {node_ids[parent]} ({parent_name}) 的雅可比矩阵:")
                        print(f"      {jac}")
        
        print("\n=== 计算图结构结束 ===\n")
        
        # 返回空的雅可比映射
        return {}
    
    # 临时替换jac_backward_pass函数
    core.jac_backward_pass = print_only_jac_backward_pass
    
    # 调用make_jac函数，它会使用我们的打印函数
    jac, val = make_jac(foo, argnum=[0,1,2])(x, y, z)
    
    # 恢复原始函数
    core.jac_backward_pass = original_jac_backward_pass
    
    return jac, val

# 打印计算图结构
jac, val = print_graph_from_jac()

# 再次计算雅可比矩阵（使用原始函数）
jac, val = make_jac(foo, argnum=[0,1,2])(x, y, z)
print(jac, val)