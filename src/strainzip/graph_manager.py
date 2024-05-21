from collections import defaultdict

import numpy as np


def assert_vertex_unfiltered(graph, v):
    vp, inv = graph.get_vertex_filter()
    assert vp.a[v] == (not inv)


def assert_all_vertices_unfiltered(graph, vs):
    vp, inv = graph.get_vertex_filter()
    assert (vp.a[vs] == (not inv)).all()


def mediate_batch_unzipping_conflicts(unmodified_path_splits):
    path_list = []
    child_update = {}
    left_update = defaultdict(list)
    right_update = defaultdict(list)
    all_parents = []
    for parent, children, child_paths in unmodified_path_splits:
        all_parents.append(parent)
        for child, (left, right) in zip(children, child_paths):
            # print(f"Adding info for {left} -> {parent}/{child} -> {right}")
            path_list.append((left, parent, right))
            child_update[(left, parent, right)] = child
            left_update[(parent, right)].append(child)
            right_update[(left, parent)].append(child)
    # print("Mapping from paths to children:", child_update)
    # print("Mapping from left edges to target children:", left_update)
    # print("Mapping from right edges to source children:", right_update)

    edge_list = []
    for left, parent, right in path_list:
        child = child_update[(left, parent, right)]
        # print(f"Adding edges for {left} -> {parent}/{child} -> {right}")
        if (left, parent) in left_update:
            left_list = left_update[(left, parent)]
            # print(
            #     f"{left} (in {left} -> {parent})"
            #     f"needs to be replace with {left_list}"
            # )
        else:
            left_list = [left]
        if (parent, right) in right_update:
            right_list = right_update[(parent, right)]
            # print(
            #     f"{right} (in {parent} -> {right}) "
            #     f" needs to be replace with {right_list}"
            # )
        else:
            right_list = [right]

        for left in left_list:
            edge_list.append((left, child))
        for right in right_list:
            edge_list.append((child, right))

    # FIXME: WORKHERE (2024-05-21)
    all_vertices_in_edges = [left for left, _ in edge_list] + [
        right for _, right in edge_list
    ]
    assert not (set(all_parents) & set(all_vertices_in_edges))
    return list(set(edge_list))


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


class TouchedUnzipper(PropertyUnzipper):
    mutates = ["touched"]

    def unzip(self, graph, parent, children, args=()):
        kwargs = self.name_free_args(args)
        graph.vp["touched"].a[parent] = True
        graph.vp["touched"].a[children] = True


class TouchedPresser(PropertyPresser):
    mutates = ["touched"]

    def press(self, graph, parents, child, args=()):
        kwargs = self.name_free_args(args)
        graph.vp["touched"].a[parents] = True
        graph.vp["touched"].a[child] = True


# class HistoryUnzipper(PropertyUnzipper):
#     mutates = ["history"]
#     requires = []
#     free_args = ["round_i"]
#
#     def unzip(self, graph, parent, children, args):
#         kwargs = self.name_free_args(args)
#         note = kwargs['round_i']
#         parent_histories = graph.vp["history"].a[parent]
#         graph.vp["history"].a[children] = unzipped_history  # TODO: Access differently, since it's a str type property.
#         graph.vp["history"][parent] = parent_history  # TODO: For each residual vertex, mark as residual.
#
# class HistoryPresser(PropertyPresser):
#     mutates = ["history"]
#     requires = []
#     free_args = ["round_i"]
#
#     def press(self, graph, parents, child, args=()):
#         kwargs = self.name_free_args(args)
#         round = kwargs["round_i"]
#         parent_histories = graph.vp["history"].a[parents]  # TODO: Need to access differently, since this is a str.
#         graph.vp["history"][child] = path_history  # TODO: Construct a new history string based on the parents being combined.
#         graph.vp["history"][parents] = parent_history  # TODO: For each residual vertex, mark as residual.


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
        # TODO: Construct path depths faster than `np.asarray`.
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
        # TODO: Construct parent depths faster than `np.asarray`.
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
        # Implicit FilterUnzipper and FilterPresser
        self.unzippers = [FilterUnzipper()] + unzippers
        self.pressers = [FilterPresser()] + pressers
        self.unzipper_free_args = self.collect_unzipper_args()
        self.presser_free_args = self.collect_presser_args()

    def assert_vertex_unfiltered(self, graph, v):
        assert graph.vp["filter"][v]

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
        # NOTE: Caution!!! Paths built from an old graph may
        # be outdated! Unzipping them may cause problems.
        # I try to check for this with assert_vertex_unfiltered, but who knows.
        assert_vertex_unfiltered(graph, parent)
        n = len(paths)
        num_before = graph.num_vertices(ignore_filter=True)
        num_after = num_before + n
        graph.add_vertex(n)
        children = list(range(num_before, num_after))
        new_edge_list = []
        for (left, right), child in zip(paths, children):
            assert_all_vertices_unfiltered(graph, [child, left, right])
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
        return children

    def press(self, graph, parents, **kwargs):
        child = graph.num_vertices(
            ignore_filter=True
        )  # Infer new node index by size of the graph.
        graph.add_vertex()
        leftmost_parent = parents[0]
        rightmost_parent = parents[-1]
        left_list = graph.get_in_neighbors(leftmost_parent)
        right_list = graph.get_out_neighbors(rightmost_parent)
        assert_all_vertices_unfiltered(
            graph, [child] + list(left_list) + list(right_list)
        )

        # Collect new edges into and out of unitig.
        new_edge_list = []
        # Detect if unitig is a cycle:
        if (leftmost_parent in right_list) or (rightmost_parent in left_list):
            new_edge_list.append((child, child))
        # Add other incoming/outgoing edges
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
        return child

    # def burst(self, graph, parents, *kwargs):
    #     # TODO: Treatment for bubbles and tips, putting all the depth
    #     # into the context sequence.
    #     pass

    def batch_unzip(self, graph, *args):
        # Args should be a tuple: (parent, paths, kwargs)
        # NOTE: If any of the vertices named in paths are also found in any
        # of the parent entries of args, then these need to be updated, because
        # they will no longer exist after the unzip operation on that vertex.
        all_children = []
        unmodified_path_splits = []
        for parent, paths, kwargs in args:
            assert_vertex_unfiltered(graph, parent)
            n = len(paths)
            num_before = graph.num_vertices(ignore_filter=True)
            num_after = num_before + n
            graph.add_vertex(n)
            children = list(range(num_before, num_after))
            all_children.append(children)
            unmodified_path_splits.append((parent, children, paths))

            for uz in self.unzippers:
                uz.unzip(
                    graph,
                    parent,
                    children,
                    tuple(kwargs[arg] for arg in uz.free_args),
                )

        new_edge_list = mediate_batch_unzipping_conflicts(unmodified_path_splits)
        graph.add_edge_list(new_edge_list)

        return all_children

    def batch_press(self, graph, *args):
        children = []
        for parents, kwargs in args:
            children.append(self.press(graph, parents, **kwargs))
        return children

    def batch_trim(self, graph, parents):
        graph.vp.filter.a[parents] = 0
