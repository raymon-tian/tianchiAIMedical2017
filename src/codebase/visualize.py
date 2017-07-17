#coding=utf-8
# pytorch 网络结构可视化脚本
# github：https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
# demo： https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb

from graphviz import Digraph
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn
import net_res18


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    # add_nodes(var.grad_fn)
    add_nodes(var.creator)
    return dot

def test_visualize():

    data = torch.rand(1,1,128,128,128)
    target = torch.rand(1,32,32,32,3,5)
    coord = torch.rand(1,3,32,32,32)

    data = Variable(data.cuda(async=True))  # 图像数据
    target = Variable(target.cuda(async=True))  #
    coord = Variable(coord.cuda(async=True))

    config, net, loss, get_pbb = net_res18.get_model()
    checkpoint = torch.load('../checkpoints/detector.ckpt')

    params = checkpoint['state_dict']

    # convert numpy arrays to torch Variables
    for k in sorted(params.keys()):
        v = params[k]
        print k, v.size()
        params[k] = Variable(v, requires_grad=True)

    checkpoint1 = torch.load('../checkpoints/detector.ckpt')
    net.load_state_dict(checkpoint1['state_dict'])
    net = net.cuda()  # 网络设置为 GPU格式
    loss = loss.cuda()  # 损失设置为 GPU格式
    cudnn.benchmark = True  # 使用cudnn加速

    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    output = net(data, coord)
    # output[0].backward()
    print(output.size())
    loss_output = criterion(output, target)
    print(loss_output)
    loss_output.backward()
    return loss_output,params
    # make_dot(loss_output,params=params)
    # make_dot(output,params=params)
    # make_dot(loss_output)

if __name__ == '__main__':
    test_visualize()