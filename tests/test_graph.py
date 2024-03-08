import strainzip as sz
import graph_tool as gt

def test_construction():
    sz.DepthGraph(gt.Graph(), num_samples=1)
