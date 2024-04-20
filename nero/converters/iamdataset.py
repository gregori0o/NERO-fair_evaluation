from typing import List, Tuple
import os

import networkx as nx
from tqdm import tqdm
import torch
from torch_geometric.data import Data

import nero.constants as constants
import nero.tools.datasets as datasets
import nero.tools.graphs as graphs
from nero.converters.common import DatasetDescription


def separate_labels_and_attributes(graph: nx.Graph) -> nx.Graph:
    result = nx.Graph()
    result.graph['category'] = graph.graph['classes'][0]
    result.add_nodes_from(graph.nodes)
    for node in graph.nodes:
        try:
            labels = graph.nodes[node]['labels']
            for i, label in enumerate(labels):
                result.nodes[node][f'node_label_{i}'] = label
        except KeyError:
            pass
        try:
            attributes = graph.nodes[node]['attributes']
            for i, attribute in enumerate(attributes):
                result.nodes[node][f'node_attribute_{i}'] = attribute
        except KeyError:
            pass
    for edge_from, edge_to, edge_data in graph.edges(data=True):
        result.add_edge(edge_from, edge_to)
        try:
            labels = edge_data['labels']
            for i, label in enumerate(labels):
                result.edges[edge_from, edge_to][f'edge_label_{i}'] = label
        except KeyError:
            pass
        try:
            attributes = edge_data['attributes']
            for i, attribute in enumerate(attributes):
                result.edges[edge_from, edge_to][f'edge_attribute_{i}'] = attribute
        except KeyError:
            pass
    return result


def iamdataset2nx(iamdataset: List[Data]) -> List[nx.Graph]:
    dataset = []
    for og in iamdataset:
        g = nx.Graph()
        # add nodes
        for j in range(og.num_nodes):
            g.add_node(j)
        
        # add edges
        edgeb_list = []
        for x, y in zip(og.edge_index[0], og.edge_index[1]):
            x, y = x.item(), y.item()
            if ((x, y)) not in list(g.edges) and ((y, x)) not in list(g.edges):
                g.add_edge(x, y)
                edgeb_list.append(True)
            else:
                edgeb_list.append(False)

        # add node labels
        for j in range(og.num_nodes):
            g.nodes[j]['labels'] = og.x[j].tolist()

        # add node attributes
        # g.nodes[v]['attributes'] = ?
        
        # add edge labels
        for j, (x, y) in enumerate(zip(og.edge_index[0], og.edge_index[1])):
            x, y = x.item(), y.item()
            if not edgeb_list[j]:
                continue
            g.edges[(x, y)]["labels"] = og.edge_attr[j].tolist()

        # add edge attributes
        # g.edges[e]['attributes'] = ?

        # add graph label
        g.graph['classes'] = [og.y[0].item()]

        # add graph attributes
        # g.graph['targets'] = ?

        dataset.append(g)

    dataset = [separate_labels_and_attributes(graph) for graph in dataset]

    return dataset


def discover_labels_and_attributes(dataset_name: str, graph: nx.Graph) -> DatasetDescription:
    node_labels = 0
    while len(nx.get_node_attributes(graph, f"node_label_{node_labels}")) > 0:
        node_labels += 1
    edge_labels = 0
    while len(nx.get_edge_attributes(graph, f"edge_label_{edge_labels}")) > 0:
        edge_labels += 1
    node_attributes = 0
    while len(nx.get_node_attributes(graph, f"node_attribute_{node_attributes}")) > 0:
        node_attributes += 1
    edge_attributes = 0
    while len(nx.get_edge_attributes(graph, f"node_attribute_{edge_attributes}")) > 0:
        edge_attributes += 1
    return DatasetDescription(
        name=dataset_name,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )


def iamdataset2persisted(
        dataset_name: str
) -> Tuple[List[datasets.PersistedClassificationSample], List[int], DatasetDescription]:
    iamdataset = torch.load(os.path.join(constants.DOWNLOADS_DIR, dataset_name, 'data.pt'))
    dataset = iamdataset2nx(iamdataset)
    iamdataset_description = discover_labels_and_attributes(dataset_name, dataset[0])
    target_classes = [graph.graph['category'] for graph in dataset]
    converted_dataset = [
        graphs.edges_into_nodes(graph)
        for graph in tqdm(dataset, desc="Converting to a bipartite form")
    ]
    converted_dataset = [
        datasets.create_persisted_sample(graph, dataset_name, i, 'category')
        for i, graph in enumerate(tqdm(converted_dataset, desc="Creating persisted samples"))
    ]
    return converted_dataset, target_classes, iamdataset_description
