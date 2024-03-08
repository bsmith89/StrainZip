import graph_tool as gt

import strainzip as sz


def test_construction():
    sz.DepthGraph(gt.Graph(), num_samples=1)
