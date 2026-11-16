from collections import defaultdict

class GraphTool:
    def __init__(self):
        self.graph = defaultdict(list)  # Node -> list of connected nodes

    def add_link(self, source, target):
        """Add a bidirectional link between nodes."""
        self.graph[source].append(target)
        self.graph[target].append(source)

    def get_links(self, node):
        """Retrieve all nodes linked to a given node."""
        return self.graph.get(node, [])

    def get_all_nodes(self):
        """Get all unique nodes in the graph."""
        return list(self.graph.keys())