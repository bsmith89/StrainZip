import numpy as np


class PropertyUnzipper:
    mutates = []
    requires = []
    free_args = []

    def __init__(self):
        pass

    def name_free_args(self, args):
        assert len(args) == len(self.free_args)
        return {k: v for k, v in zip(self.free_args, args)}

    def validate(self, graph):
        for prop in self.mutates + self.requires:
            assert prop in graph.vp

    def unzip(self, graph, parent, children, args):
        raise NotImplementedError

    def batch_unzip(self, graph, *args):
        for parent, children, _args in args:
            self.unzip(graph, parent, children, _args)


class PropertyPresser:
    mutates = []
    requires = []
    free_args = []

    def __init__(self):
        pass

    def name_free_args(self, args):
        assert len(args) == len(self.free_args)
        return {k: v for k, v in zip(self.free_args, args)}

    def validate(self, graph):
        for prop in self.mutates + self.requires:
            assert prop in graph.vp

    def press(self, graph, parents, child, args):
        raise NotImplementedError

    def batch_press(self, graph, *args):
        for parents, child, _args in args:
            self.press(graph, parents, child, _args)


class FilterUnzipper(PropertyUnzipper):
    mutates = ["filter"]
    requires = []
    free_args = []

    def unzip(self, graph, parent, children, args=()):
        kwargs = self.name_free_args(args)
        graph.vp["filter"][parent] = False
        graph.vp["filter"].a[children] = True


class FilterPresser(PropertyPresser):
    mutates = ["filter"]
    requires = []
    free_args = []

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        graph.vp["filter"].a[parents] = False
        graph.vp["filter"][child] = True


class LengthUnzipper(PropertyUnzipper):
    mutates = ["length"]
    requires = []
    free_args = []

    def unzip(self, graph, parent, children, args=()):
        kwargs = self.name_free_args(args)
        parent_val = graph.vp["length"][parent]
        graph.vp["length"].a[children] = parent_val


class LengthPresser(PropertyPresser):
    mutates = ["length"]
    requires = []
    free_args = []

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        parent_vals = graph.vp["length"].a[parents]
        graph.vp["length"][child] = parent_vals.sum()


class SequenceUnzipper(PropertyUnzipper):
    mutates = ["sequence"]
    requires = []
    free_args = []

    def unzip(self, graph, parent, children, args=()):
        kwargs = self.name_free_args(args)
        parent_val = graph.vp["sequence"][parent]
        for child in children:
            graph.vp["sequence"][child] = parent_val


class SequencePresser(PropertyPresser):
    mutates = ["sequence"]
    requires = []
    free_args = []

    def __init__(self, sep=","):
        super().__init__()
        self.sep = ","

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        parent_vals = [graph.vp["sequence"][p] for p in parents]
        graph.vp["sequence"][child] = self.sep.join(parent_vals)


class ScalarDepthUnzipper(PropertyUnzipper):
    mutates = ["depth"]
    requires = []
    free_args = ["path_depths"]

    def unzip(self, graph, parent, children, args):
        kwargs = self.name_free_args(args)
        parent_depth = graph.vp["depth"].a[parent]
        path_depths = np.asarray(kwargs["path_depths"])
        graph.vp["depth"].a[children] = path_depths
        graph.vp["depth"][parent] = parent_depth - path_depths.sum()


class ScalarDepthPresser(PropertyPresser):
    mutates = ["depth"]
    requires = ["length"]
    free_args = []

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        parents_depth = graph.vp["depth"].a[parents]
        parents_length = graph.vp["length"].a[parents]
        weighted_mean_depth = (
            parents_depth * parents_length
        ).sum() / parents_length.sum()
        graph.vp["depth"][child] = weighted_mean_depth
        graph.vp["depth"].a[parents] = parents_depth - weighted_mean_depth


class VectorDepthUnzipper(PropertyUnzipper):
    mutates = ["depth"]
    requires = []
    free_args = ["path_depths"]

    def __init__(self):
        super().__init__()

    def unzip(self, graph, parent, children, args):
        kwargs = self.name_free_args(args)
        path_depths = np.asarray(kwargs["path_depths"])
        parent_depth = graph.vp["depth"][parent]
        updated_parent_depth = parent_depth - path_depths.sum(0)

        for child, _path_depth in zip(children, path_depths):
            graph.vp["depth"][child] = _path_depth

        graph.vp["depth"][parent] = updated_parent_depth


class VectorDepthPresser(PropertyPresser):
    mutates = ["depth"]
    requires = ["length"]
    free_args = []

    def __init__(self):
        super().__init__()

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        nparents = len(parents)
        parent_depths = np.asarray([graph.vp["depth"][p] for p in parents])
        parent_lengths = graph.vp["length"].a[parents]
        weighted_mean_depth = (
            parent_depths * parent_lengths.reshape((nparents, 1))
        ).sum(0) / parent_lengths.sum()

        graph.vp["depth"][child] = weighted_mean_depth
        for p, _parent_depth in zip(parents, parent_depths):
            graph.vp["depth"][p] = _parent_depth - weighted_mean_depth


class PositionUnzipper(PropertyUnzipper):
    mutates = ["xyposition"]
    requires = []
    free_args = []

    def __init__(self, offset=(0.1, 0.1)):
        self.xoffset, self.yoffset = offset

    def unzip(self, graph, parent, children, args=()):
        kwargs = self.name_free_args(args)
        parent_pos = graph.vp["xyposition"][parent]  # .reshape((2, 1))

        num_children = len(children)
        xoffsets = (
            np.linspace(-0.5, 0.5, num=num_children) * self.xoffset * num_children
        )
        yoffsets = (
            np.linspace(-0.5, 0.5, num=num_children) * self.yoffset * num_children
        )
        offsets = np.stack([xoffsets, yoffsets])
        child_pos = (offsets.T + parent_pos).T

        for i, child in enumerate(children):
            graph.vp["xyposition"][child] = child_pos[:, i]


class PositionPresser(PropertyPresser):
    mutates = ["xyposition"]
    requires = ["length"]
    free_args = []

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        nparents = len(parents)
        parent_pos = np.asarray([graph.vp["xyposition"][p] for p in parents])
        parent_lengths = graph.vp["length"].a[parents]
        weighted_mean_pos = (parent_pos * parent_lengths.reshape((nparents, 1))).sum(
            0
        ) / parent_lengths.sum()

        graph.vp["xyposition"][child] = weighted_mean_pos


class GraphManager:
    def __init__(self, unzippers=[], pressers=[]):
        self.unzippers = unzippers
        self.pressers = pressers
        self.unzipper_free_args = self.collect_unzipper_args()
        self.presser_free_args = self.collect_presser_args()

    def collect_unzipper_args(self):
        free_args = set()
        for unzipper in self.unzippers:
            free_args |= set(unzipper.free_args)
        return list(free_args)

    def collect_presser_args(self):
        free_args = set()
        for presser in self.pressers:
            free_args |= set(presser.free_args)
        return list(free_args)

    def validate(self, graph):
        all_unzipper_targets = []
        all_presser_targets = []
        all_requires = []
        for uz in self.unzippers:
            all_unzipper_targets.extend(uz.mutates)
            all_requires.extend(uz.requires)
        for pr in self.pressers:
            all_presser_targets.extend(pr.mutates)
            all_requires.extend(pr.requires)

        assert len(all_unzipper_targets) == len(
            set(all_unzipper_targets)
        ), "Multiple unzippers targetting a property."
        assert len(all_presser_targets) == len(
            set(all_presser_targets)
        ), "Multiple pressers targetting a property."

        all_targets = set(all_unzipper_targets + all_presser_targets)

        for prop in all_targets:
            assert (
                prop in graph.vp.keys()
            ), f"Target property '{prop}' missing from graph vertex properties."

        for prop in all_requires:
            assert (
                prop in graph.vp.keys()
            ), f"Required property '{prop}' missing from graph vertex properties."

        for prop in graph.vp.keys():
            assert (
                prop in all_unzipper_targets
            ), f"Graph vertex property '{prop}' missing from unzipper targets."
            assert (
                prop in all_presser_targets
            ), f"Graph vertex property '{prop}' missing from presser targets."

    def unzip(self, graph, parent, paths, **kwargs):
        n = len(paths)
        num_before = graph.num_vertices(ignore_filter=True)
        num_after = num_before + n
        graph.add_vertex(n)
        children = list(range(num_before, num_after))
        new_edge_list = []
        for (left, right), child in zip(paths, children):
            new_edge_list.append((left, child))
            new_edge_list.append((child, right))
        graph.add_edge_list(new_edge_list)

        for uz in self.unzippers:
            uz.unzip(
                graph,
                parent,
                children,
                tuple(kwargs[arg] for arg in uz.free_args),
            )

    def press(self, graph, parents, **kwargs):
        child = graph.num_vertices(
            ignore_filter=True
        )  # Infer new node index by size of the graph.
        graph.add_vertex()
        leftmost_parent = parents[0]
        rightmost_parent = parents[-1]
        left_list = graph.get_in_neighbors(leftmost_parent)
        right_list = graph.get_out_neighbors(rightmost_parent)
        new_edge_list = []
        for left in left_list:
            new_edge_list.append((left, child))
        for right in right_list:
            new_edge_list.append((child, right))
        graph.add_edge_list(new_edge_list)

        for pr in self.pressers:
            pr.press(
                graph,
                parents,
                child,
                tuple(kwargs[arg] for arg in pr.free_args),
            )

    def batch_unzip(self, graph, *args):
        for parent, paths, kwargs in args:
            self.unzip(graph, parent, paths, **kwargs)

    def batch_press(self, graph, *args):
        for parents, kwargs in args:
            self.press(graph, parents, **kwargs)