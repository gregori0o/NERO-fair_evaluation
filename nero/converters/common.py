from dataclasses import dataclass


@dataclass
class DatasetDescription:
    name: str
    node_labels: int
    edge_labels: int
    node_attributes: int
    edge_attributes: int
