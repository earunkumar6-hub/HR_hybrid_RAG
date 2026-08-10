"""Renders the backend's node-link knowledge-graph JSON as an interactive
pyvis network, returned as HTML for st.components.v1.html()."""
from pyvis.network import Network

# Nodes are coloured by entity_type (as extracted by the LLM into the
# knowledge graph — person/team/system/policy/process/project/...),
# not by any business taxonomy. Unrecognised/blank types fall back to grey.
ENTITY_TYPE_COLORS = {
    "person": "#F76707",
    "team": "#12B886",
    "system": "#4C6EF5",
    "policy": "#AE3EC9",
    "process": "#E8590C",
    "project": "#1098AD",
    "regulation": "#C92A2A",
    "term": "#868E96",
}
DOCUMENT_COLOR = "#343A40"
DEFAULT_COLOR = "#868E96"


def render_graph_html(graph_data: dict, height: str = "600px") -> str:
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="white", notebook=False, directed=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

    for node in graph_data.get("nodes", []):
        is_document = node["type"] == "document"
        color = (
            DOCUMENT_COLOR
            if is_document
            else ENTITY_TYPE_COLORS.get((node.get("entity_type") or "").lower(), DEFAULT_COLOR)
        )
        shape = "box" if is_document else "dot"
        size = 22 if is_document else 14
        title = f"{node['label']} ({node.get('entity_type') or node['type']})"
        net.add_node(node["id"], label=node["label"], title=title, color=color, shape=shape, size=size)

    for edge in graph_data.get("edges", []):
        net.add_edge(edge["source"], edge["target"], title=edge.get("relation", ""))

    return net.generate_html(notebook=False)
