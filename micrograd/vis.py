from graphviz import Digraph

# TODO: Can we also apply BFS here? Is it related to backprop
def trace(root):
    """ Builds a set of all nodes and edges in a graph by applying depth-first search (DFS)"""
    nodes, edges = set(), set()
    
    def dfs(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v)) # Is it directed?
                dfs(child)
    
    dfs(root)
    return nodes, edges
    

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR: left to right

    def make_node_id(node, op=''):
        return str(id(node)) + op

    nodes, edges = trace(root)
    for v in nodes:
        uid = make_node_id(v)
        # For any value in the graph, create a rectangular (record) node for it
        dot.node(name=uid, label="{%s | data=%.4f | grad=%.4f}" % (v.label, v.data, v.grad), shape='record')
        if v._op:
            # If this value is a result of some operation, create an op node and connect it
            uid_op = make_node_id(v, v._op)
            dot.node(name=uid_op, label=v._op)
            dot.edge(uid_op, uid)
    
    for u, v in edges:
        # Connect nides with operations
        dot.edge(make_node_id(u), make_node_id(v, v._op))

    return dot
