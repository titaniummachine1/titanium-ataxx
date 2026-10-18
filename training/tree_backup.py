#!/usr/bin/env python3
"""Tree-backup training (user design): the whole recorded search becomes one
differentiable graph. Static evals are leaves of the graph; backed-up values
flow through (soft) max backups; loss vs deep targets. No re-search needed —
the tree is data. Hard max = blame along PV only; soft backup (temp) =
distributed blame, the cooperative setting.
"""
import torch


def soft_backup(children_vals, temp=50.0):
    """Differentiable max over child values (stm-local frame)."""
    if len(children_vals) == 1:
        return children_vals[0]
    stacked = torch.stack(children_vals)
    return temp * torch.logsumexp(stacked / temp, dim=0)


def backup_tree(statics, children, temp=50.0):
    """statics: {node: tensor scalar}; children: {node: [child...]} (stm-local
    frame: parent value = backup over -child values... here caller passes
    already-negated child contributions). Returns {node: value} for all."""
    vals = dict(statics)
    order = list(toposort(children))
    for n in order:
        if n in children and children[n]:
            vals[n] = soft_backup([vals[c] for c in children[n]], temp)
    return vals


def toposort(children):
    seen = set()
    out = []

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for c in children.get(n, []):
            visit(c)
        out.append(n)

    for n in children:
        visit(n)
    return out


def tree_loss(statics, children, root, root_target, temp=50.0, align_w=0.1):
    """Root anchored to deep target + every node aligned to its soft backup
    (recursive consistency = the whole tree cooperates)."""
    vals = backup_tree(statics, children, temp)
    loss = (vals[root] - root_target) ** 2
    for n, chs in children.items():
        if chs:
            loss = loss + align_w * (statics[n] - vals[n].detach()) ** 2
    return loss


def _test():
    torch.manual_seed(0)
    # synthetic tree: root(0) -> {1, 2}; 1 -> {3, 4}; 2 -> {5}. stm-local max.
    children = {0: [1, 2], 1: [3, 4], 2: [5]}
    w = torch.nn.Parameter(torch.randn(6))
    opt = torch.optim.Adam([w], lr=0.05)
    l0 = None
    for step in range(200):
        opt.zero_grad()
        statics = {i: w[i] for i in range(6)}
        loss = tree_loss(statics, children, root=0,
                         root_target=torch.tensor(3.0), temp=0.5)
        if l0 is None:
            l0 = loss.item()
        loss.backward()
        opt.step()
    vals = backup_tree({i: w[i].detach() for i in range(6)}, children, temp=0.5)
    print('loss %.4f -> %.4f' % (l0, loss.item()))
    print('root backed %.3f (target 3.0)' % vals[0].item())
    assert loss.item() < l0 * 0.1, 'no learning'
    assert abs(vals[0].item() - 3.0) < 0.3, 'root did not converge'
    print('TREE-BACKUP OK')


if __name__ == '__main__':
    _test()
