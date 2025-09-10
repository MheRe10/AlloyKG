import networkx as nx
from pyvis.network import Network
import os

# Load the graphml file
try:
    G = nx.read_graphml("../data/end_to_end/rag_storage/graph_chunk_entity_relation.graphml")
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
except Exception as e:
    print(f"Error loading graph: {e}")
    exit(1)

# Convert to PyVis for visualization
net = Network(notebook=False, width="100%", height="800px", directed=True)

# Set some options for better visualization
net.set_options("""
var options = {
  "nodes": {
    "color": {
      "border": "rgba(0,0,0,1)",
      "background": "rgba(97,195,238,1)"
    },
    "font": {"size": 12}
  },
  "edges": {
    "color": {"inherit": true},
    "smooth": false
  },
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 100}
  }
}
""")

try:
    net.from_nx(G)
    
    # Save HTML file directly instead of using show()
    html_file = "knowledge_graph.html"
    net.write_html(html_file)
    print(f"Visualization saved to {html_file}")
    
    # Try to open in default browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(html_file)}")
    
except Exception as e:
    print(f"Error creating visualization: {e}")
    print("Trying alternative method...")
    
    # Alternative: create a simple HTML visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph</title>
    </head>
    <body>
        <h1>Knowledge Graph Statistics</h1>
        <p>Nodes: {G.number_of_nodes()}</p>
        <p>Edges: {G.number_of_edges()}</p>
        <p>Graph could not be visualized due to library issues.</p>
    </body>
    </html>
    """
    
    with open("knowledge_graph_simple.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Simple statistics saved to knowledge_graph_simple.html")
