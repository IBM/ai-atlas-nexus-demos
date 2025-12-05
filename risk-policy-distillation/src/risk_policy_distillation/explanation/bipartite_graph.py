import copy
import logging

import numpy as np

logger = logging.getLogger('logger')


class Node:

    def __init__(self, id, value, probability=1.0, subnodes=[]):
        self.id = id
        self.value = value
        self.probability = probability
        self.subnodes = subnodes

        self.num_subnodes = sum([s.num_subnodes for s in self.subnodes]) + len(subnodes)

    def get_importance(self, n):
        # TODO: this should be normalized to [0, 1]
        return self.num_subnodes


class Edge:

    def __init__(self, id, source, target, source_side=0):
        self.id = id
        self.source = source
        self.target = target
        self.source_side = source_side


class BipartiteGraph:


    def __init__(self, labels):
        """
        Bipartite graph representing a set of local explanations through concepts
        :param labels: List of possible labels in the task
        """
        self.k = len(labels)
        self.labels = labels

        self.nodes = {l: [] for l in labels}
        self.start_sizes = {l: 0 for l in labels}
        self.counts = {l: 0 for l in labels}

        self.edges = []

    def add_node(self, node, label=0):
        """
        Adds a new node to the graph
        :param node: New node
        :param label: Partition to add the node to
        :return:
        """
        self.nodes[label].append(node)
        self.counts[label] += 1

    def add_edge(self, edge):
        """
        Adds a new edge to the graph
        :param edge: New edge
        :return:
        """
        self.edges.append(edge)

    def get_edge_nodes(self, e):
        """
        Returns the source and target nodes of edge e
        :param e: Edge
        :return: source and target nodes for edge e
        """
        source = [n for n in self.nodes[e.source_side] if n.id == e.source][0]

        target = None
        for l, l_nodes in self.nodes.items():
            if l != e.source_side:
                for t in l_nodes:
                    if t.id == e.target:
                        return source, t

        return source, target

    def load_graph(self, connected_nodes, label=0):
        """
        Creates the graph from a list of connected concept
        :param connected_nodes: A list of concept pairs
        :param label: A partition to assign the nodes to
        :return:
        """
        for c in connected_nodes:
            if c[label] != 'none':
                central_node = Node(id=self.counts[label], value=c[label])
                self.add_node(central_node, label)

                for l in self.labels:
                    if l != label and c[l] != 'none':
                        node = Node(id=self.counts[l], value=c[l])
                        self.add_node(node, l)
                        edge = Edge(len(self.edges), central_node.id, node.id, source_side=label)
                        self.add_edge(edge)

        self.start_sizes[label] = len(self.nodes[label])

        logger.info('Loaded the following graph:\n\tLabels = {} Sizes = {} Number of edges = {}'.format(self.nodes.keys(),
                                                                                                        [len(self.nodes[k]) for k in self.labels],
                                                                                                        len(self.edges)))

    def merge_nodes(self, node_ids, new_label, probability, side, cleanup=False):
        """
        Merges nodes with ids in node_ids into a new node with new_label as label
        :param node_ids: ids of nodes to be merged
        :param new_label: new label to be assigned
        :param probability: FactReasoner-type probability of entailment of the new node on the merged nodes
        :param side: partition of the graph where the merged node are located
        :return: A new node with new_label as label
        """
        logger.info('\n\t\t\tMerging {} nodes on {} side.'.format(len(node_ids), side))
        merging = self.nodes[side]

        old_nodes = copy.deepcopy([n for n in merging if n.id in node_ids])

        old_edges_source = [e for e in self.edges if e.source in node_ids and e.source_side == side]
        old_edges_target = [e for e in self.edges if e.target in node_ids and e.source_side != side]

        for n in [n for n in merging if n.id in node_ids]:
            self.nodes[side].remove(n)

        for e in old_edges_source + old_edges_target:
            self.edges.remove(e)

        new_node = Node(id=self.counts[side], value=new_label, probability=probability, subnodes=[] if cleanup else old_nodes) # if just cleaning up and merging exact same nodes -- don't add them to the subnodes to impact the importance
        logger.info('\t\tAdded a node: id = label = {}, probability = {}, num of subnodes = {}'.format(new_node.id, new_label, probability, len(old_nodes)))

        self.add_node(new_node, side)

        for e in old_edges_source:
            source = new_node.id
            target = e.target

            exists = len([e for e in self.edges if (e.source == source) and (e.target == target)])
            # if an edge source and target does not exist then add it
            if not exists:
                edge = Edge(id=len(self.edges), source=source, target=target, source_side=e.source_side)
                self.edges.append(edge)

        for e in old_edges_target:
            source = e.source
            target = new_node.id

            exists = len([e for e in self.edges if (e.source == source) and (e.target == target)])
            # if an edge source and target does not exist then add it
            if not exists:
                edge = Edge(id=len(self.edges), source=source, target=target, source_side=e.source_side)
                self.edges.append(edge)

    def size(self):
        """
        :return: Total number of nodes in the graph
        """
        return sum([len(nodes) for nodes in self.nodes.values()])

    def get_nodes(self, side=0):
        """
        :param side: Graph partition
        :return: Nodes in side partition
        """
        return self.nodes[side]

    def get_expl(self):
        """
        Generates an ordered set of global rules to serve as an explanation
        :return: a list of rules ordered from most to least important
        """
        rules = []
        for l in self.labels:
            rules += self.collect_rules(self.nodes[l], l)

        importances = [r.importance for r in rules]
        sorted = list(np.argsort(importances))

        reordered_rules = []
        for i in sorted:
            reordered_rules.append(rules[i])

        return reordered_rules

    def collect_rules(self, nodes, side=0):
        """
        Collects rules that support a specific decision denoted by the side parameter
        :param nodes: Nodes in the partition side
        :param side: partition
        :return: a list of rules supporting the decision side
        """
        rules = []
        for n in nodes:
            source_for_edges = [e for e in self.edges if n.id == e.source and e.source_side==side]
            if len(source_for_edges):
                despites = []
                for e in source_for_edges:
                    source, target = self.get_edge_nodes(e)
                    if target.num_subnodes:
                        despites.append(target.value)

                size = self.start_sizes[side]
                importance = n.get_importance(size)
                prediction = side

                r = Rule(argument_because=n.value, argument_despite=despites, prediction=prediction, importance=importance)
                rules.append(r)

        return rules

    def print(self):
        for label, nodes in self.nodes.items():
            logger.info('Label = {}'.format(label))
            for n in nodes:
                logger.info(f'\t{n.id}: {n.value}')


class Rule:

    def __init__(self, argument_because, argument_despite, prediction, importance):
        self.argument_because = argument_because
        self.argument_despite = argument_despite
        self.prediction = prediction
        self.importance = importance

    def print(self):
        return '({} | {}) Pred: {} Importance:{}'.format(self.argument_because, ','.join(self.argument_despite), self.prediction, self.importance)