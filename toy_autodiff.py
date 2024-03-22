#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:45:17 2024

@author: lixiao
"""

import numpy as np

class Node(object):
  def __init__(self, value = None, name = None):
    self.name = name
    self.value = value
    self.inputs = []
    self.grad = 0
    self.output_grad = 0
    
  def __mul__(self, other):
    return mul_op(self, other)
  
  def __add__(self, other):
    return add_op(self, other)
  
  def __str__(self):
    """Allow print to display node name.""" 
    return self.name + ' Op: ' + self.op.__str__()
  
  __radd__ = __add__
  __rmul__ = __mul__
  __repr__ = __str__

Operated_type  = Node | int | float | list
topo_forward = []
class Op(object):
  """Op represents operations performed on nodes."""
  
  def __call__(self):
    new_node = Node()
    new_node.op = self
    topo_forward.append(new_node)
    return new_node
  
  def compute(self, node: Node, input_vals: Operated_type):
    raise NotImplementedError

  def gradient(self, node: Node, output_grad: Operated_type):
    raise NotImplementedError

class MulOp(Op):
  def __call__(self, a: Node, b : Operated_type):
    new_node = Op.__call__(self)
    if isinstance (b, Node):
      new_node.name = "(%s*%s)" % (a.name, b.name)
    else:
      new_node.name = "(%s*%s)" % (a.name, b)
    new_node.inputs.extend([a, b])
    return new_node
  
  def __str__(self):
    return 'Mul'
  __repr__ = __str__
  
  def compute(self, node: Node, inputs: Operated_type):
    a = inputs[0].value
    b = inputs[1]
    if isinstance (inputs[1], Node):
      b = inputs[1].value
    return a * b
  
  def gradient(self, node: Node, output_grad: Operated_type):
    inputs = node.inputs
    a = inputs[0].value
    b = inputs[1]
    if isinstance (inputs[1], Node):
      b = inputs[1].value
      return [b*output_grad, a*output_grad]

    return [b*output_grad]
    
class AddOp(Op):
  def __call__(self, a: Node, b : Operated_type):
    new_node = Op.__call__(self)
    if isinstance (b, Node):
      new_node.name = "(%s+%s)" % (a.name, b.name)
    else:
      new_node.name = "(%s+%s)" % (a.name, b)
    new_node.inputs.extend([a, b])
    return new_node
  
  def __str__(self):
    return 'Add'
  __repr__ = __str__
  
  def compute(self, node: Node, inputs: Operated_type):
    a = inputs[0].value
    b = inputs[1]
    if isinstance (inputs[1], Node):
      b = inputs[1].value
    return a + b
  
  def gradient(self, node: Node, output_grad: Operated_type):
    inputs = node.inputs
    if isinstance (inputs[1], Node):
      return [output_grad, output_grad]
    
    return [output_grad]

class Variable(Op):
  def __call__(self, value, name):
    new_node = Op.__call__(self)
    new_node.name = "Var[%s]" % (name)
    new_node.value = value
    return new_node
  
  def __str__(self):
    return 'Variable'
  __repr__ = __str__
  
  def compute(self, node: Node, input_vals: Operated_type):
    return node.value
  
  def gradient(self, node: Node, output_grad: Operated_type):
    return [output_grad]

mul_op = MulOp()
add_op = AddOp()
Var    = Variable()

class Executor:
  def __init__(self, root):
    self.topo_list = self.__topo_sort([root])
    self.r_topo_list = list(reversed(self.topo_list))
    self.root = root
    
  def gradient(self):
    self.__forward_pass()
    self.__backward_pass()
  
  def __topo_sort_dfs(self, node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
      return
    if not isinstance(node, Node):
      return
    
    visited.append(node)
    for n in node.inputs:
      self.__topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)
  
  def __topo_sort(self, root):
    visited = []
    topo_order = []
    for node in root:
        self.__topo_sort_dfs(node, visited, topo_order)
    return topo_order

  def __forward_pass(self):
    for node in self.topo_list:
      node.value = node.op.compute(node, node.inputs)
      
  def __backward_pass(self):
    self.r_topo_list[0].output_grad = 1 # dy/dy
    for node in self.r_topo_list:
      grads = node.op.gradient(node, node.output_grad)
      for i, g in zip(node.inputs, grads):
        if isinstance(i, Node):
          i.grad += g
          i.output_grad = g

def __test_topo_sort():
  print('\n\n ------ Test: Topology Sort ------\n')
  x1 = Var(2., 'x1')
  x2 = Var(3., 'x2')
  y = x1*x2 + x2*4.3 + 30
  ex = Executor(y)
  tp_lst = ex.topo_list
  print(tp_lst, '\n')
  print(topo_forward)

def __test_FB_pass():
  print('\n\n ------ Test: Forward/Backward Pass ------\n')
  x1 = Var(2., 'x1')
  x2 = Var(3., 'x2')
  y = x1*x2 + x2*4.3 + 30


  print(y)
  print('\n======= Forward Pass ================')
  for node in topo_forward:
    print('Inputs: ', node.inputs)
    node.value = node.op.compute(node, node.inputs)
    print('Node: ', node, ' value: ', node.value, '\n')
    
    
  print('\n======= backward Pass ================')
  topo_backward = reversed(topo_forward)
  nodes = list(topo_backward)
  nodes[0].output_grad = 1
  def layer_print(node):
    grads = node.op.gradient(node, node.output_grad)
    for i, g in zip(node.inputs, grads):
      if isinstance(i, Node):
        i.grad += g
        i.output_grad = g
    print('Inputs: ', node.inputs)
    print('Node: ', node,', grad:', node.grad, '\n')

  layer_print(nodes[0])
  layer_print(nodes[1])
  layer_print(nodes[2])
  layer_print(nodes[3])
  layer_print(nodes[4])

  print("x1.grad=" , x1.grad)
  print("x2.grad=" , x2.grad)

def __test_ad():
  print('\n\n ------ Test: Automatic Differentiation ------\n')
  def func1(x1, x2):
    return x1*x2 + x2*4.3 + 20
  
  def finite_diff_x1(x1, x2):
    return (func1(x1+1e-5, x2) - func1(x1, x2))/1e-5

  def finite_diff_x2(x1, x2):
    return (func1(x1, x2+1e-5) - func1(x1, x2))/1e-5
  
  x1 = Var(2., 'x1')
  x2 = Var(3., 'x2')
  y = func1(x1, x2)
  ex = Executor(y)
  ex.gradient()
  print("x1.grad=" , x1.grad)
  print("x2.grad=" , x2.grad)
  print("x1.diff=" , finite_diff_x1(2, 3))
  print("x2.diff=" , finite_diff_x2(2, 3))
  
  print('\n')
  x1 = Var(2.*np.ones(3), 'x1')
  x2 = Var(3.*np.ones(3), 'x2')
  y = func1(x1, x2)
  ex = Executor(y)
  ex.gradient()
  print("x1.grad=" , x1.grad)
  print("x2.grad=" , x2.grad)
  print("x1.diff=" , finite_diff_x1(2.*np.ones(3), 3.*np.ones(3)))
  print("x2.diff=" , finite_diff_x2(2.*np.ones(3), 3.*np.ones(3)))
  
  print('\n')
  x1 = Var(np.array([1,2,3]), 'x1')
  x2 = Var(np.array([2,2,6]), 'x2')
  y = func1(x1, x2)
  ex = Executor(y)
  ex.gradient()
  print("x1.grad=" , x1.grad)
  print("x2.grad=" , x2.grad)
  print("x1.diff=" , finite_diff_x1(np.array([1,2,3]), np.array([2,2,6])))
  print("x2.diff=" , finite_diff_x2(np.array([1,2,3]), np.array([2,2,6])))
  
''' main '''

### test ###

__test_topo_sort()
__test_FB_pass()
__test_ad()

############
































































