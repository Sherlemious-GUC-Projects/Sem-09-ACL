import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Any

from utils.types import ContextChunk, IntentType


def _extract_graph_data_from_context(
    context_chunks: List[ContextChunk]
) -> Dict[str, Any]:
    """
    Extracts nodes and edges from ContextChunk metadata based on intent.
    This is a heuristic approach given the flat metadata structure.
    """
    nodes = {}  # {node_id: {label: ..., type: ...}}
    edges = []  # [{source: node_id, target: node_id, label: ...}]

    for chunk in context_chunks:
        metadata = chunk.metadata
        intent = next((t for t in IntentType if t.value in chunk.id), None)

        if not intent:
            # Fallback if intent cannot be inferred from ID (less reliable)
            if "flight" in metadata and "origin" in metadata and "dest" in metadata:
                intent = IntentType.FLIGHT_SEARCH
            elif "flight" in metadata and "avg_delay" in metadata:
                intent = IntentType.DELAY_ANALYSIS
            elif "generation" in metadata and "avg_food" in metadata:
                intent = IntentType.SATISFACTION_METRICS
            elif "route" in metadata and "flights" in metadata:
                intent = IntentType.ROUTE_STATS
            elif "loyalty_level" in metadata and "avg_miles" in metadata:
                intent = IntentType.LOYALTY_ANALYSIS
            elif "generation" in metadata and "journeys" in metadata:
                intent = IntentType.DEMOGRAPHIC_INSIGHTS
            elif "flight" in metadata and "feedback_count" in metadata:
                intent = IntentType.FEEDBACK_VOLUME
            elif "fleet" in metadata and "avg_delay" in metadata:
                intent = IntentType.FLEET_PERFORMANCE
            elif "cabin_class" in metadata and "journeys" in metadata:
                intent = IntentType.CABIN_CLASS_STATS
            elif "type" in metadata and "journeys" in metadata:
                intent = IntentType.CONNECTION_STATS
            elif "airport" in metadata and "flights" in metadata:
                intent = IntentType.AIRPORT_STATS
            else:
                st.warning(f"Could not infer intent for chunk: {chunk.id}")
                continue

        if intent == IntentType.FLIGHT_SEARCH:
            flight_id = f"Flight_{metadata.get('flight')}"
            origin_id = f"Airport_{metadata.get('origin')}"
            dest_id = f"Airport_{metadata.get('dest')}"
            fleet_type = metadata.get("fleet", "Unknown")

            nodes[flight_id] = {
                "label": f"Flight {metadata.get('flight')}",
                "type": "Flight",
                "details": f"Fleet: {fleet_type}",
            }
            nodes[origin_id] = {
                "label": f"Airport {metadata.get('origin')}",
                "type": "Airport",
            }
            nodes[dest_id] = {
                "label": f"Airport {metadata.get('dest')}",
                "type": "Airport",
            }

            edges.append(
                {"source": flight_id, "target": origin_id, "label": "DEPARTS_FROM"}
            )
            edges.append(
                {"source": flight_id, "target": dest_id, "label": "ARRIVES_AT"}
            )

        elif intent == IntentType.DELAY_ANALYSIS:
            flight_id = f"Flight_{metadata.get('flight')}"
            nodes[flight_id] = {
                "label": f"Flight {metadata.get('flight')}",
                "type": "Flight",
                "details": f"Avg Delay: {metadata.get('avg_delay')} min",
            }
            # No explicit edges directly from this chunk, but flight node is important

        elif intent == IntentType.SATISFACTION_METRICS:
            generation = metadata.get("generation")
            if generation:
                gen_id = f"Generation_{generation}"
                nodes[gen_id] = {
                    "label": f"Generation {generation}",
                    "type": "Generation",
                    "details": f"Avg Food Score: {metadata.get('avg_food')}",
                }

        elif intent == IntentType.ROUTE_STATS:
            route = metadata.get("route")
            if route:
                origin, dest = route.split("-")
                origin_id = f"Airport_{origin}"
                dest_id = f"Airport_{dest}"
                route_id = f"Route_{route}"

                nodes[origin_id] = {"label": f"Airport {origin}", "type": "Airport"}
                nodes[dest_id] = {"label": f"Airport {dest}", "type": "Airport"}
                nodes[route_id] = {
                    "label": f"Route {route}",
                    "type": "Route",
                    "details": f"{metadata.get('flights')} flights",
                }

                edges.append(
                    {
                        "source": origin_id,
                        "target": route_id,
                        "label": "HAS_ROUTE_START",
                    }
                )
                edges.append(
                    {"source": route_id, "target": dest_id, "label": "HAS_ROUTE_END"}
                )

        elif intent == IntentType.LOYALTY_ANALYSIS:
            loyalty_level = metadata.get("loyalty_level")
            if loyalty_level:
                level_id = f"Loyalty_{loyalty_level}"
                nodes[level_id] = {
                    "label": f"Loyalty Level {loyalty_level}",
                    "type": "Loyalty",
                    "details": f"Avg Miles: {metadata.get('avg_miles')}",
                }

        elif intent == IntentType.DEMOGRAPHIC_INSIGHTS:
            generation = metadata.get("generation")
            if generation:
                gen_id = f"Generation_{generation}"
                nodes[gen_id] = {
                    "label": f"Generation {generation}",
                    "type": "Generation",
                    "details": f"Journeys: {metadata.get('journeys')}, Avg Food: {metadata.get('avg_food')}",
                }

        elif intent == IntentType.FEEDBACK_VOLUME:
            flight_id = f"Flight_{metadata.get('flight')}"
            nodes[flight_id] = {
                "label": f"Flight {metadata.get('flight')}",
                "type": "Flight",
                "details": f"Feedback: {metadata.get('feedback_count')}",
            }

        elif intent == IntentType.FLEET_PERFORMANCE:
            fleet_type = metadata.get("fleet")
            if fleet_type:
                fleet_id = f"Fleet_{fleet_type}"
                nodes[fleet_id] = {
                    "label": f"Fleet {fleet_type}",
                    "type": "Fleet",
                    "details": f"Avg Delay: {metadata.get('avg_delay')}, Avg Food: {metadata.get('avg_food')}",
                }

        elif intent == IntentType.CABIN_CLASS_STATS:
            cabin_class = metadata.get("cabin_class")
            if cabin_class:
                cabin_id = f"Cabin_{cabin_class}"
                nodes[cabin_id] = {
                    "label": f"Cabin Class {cabin_class}",
                    "type": "Cabin Class",
                    "details": f"Journeys: {metadata.get('journeys')}",
                }

        elif intent == IntentType.CONNECTION_STATS:
            conn_type = metadata.get("type")
            if conn_type:
                conn_id = f"Connection_{conn_type.replace(' ', '_')}"
                nodes[conn_id] = {
                    "label": f"{conn_type} Connection",
                    "type": "Connection",
                    "details": f"Journeys: {metadata.get('journeys')}, Avg Delay: {metadata.get('avg_delay')}",
                }

        elif intent == IntentType.AIRPORT_STATS:
            airport = metadata.get("airport")
            if airport:
                airport_id = f"Airport_{airport}"
                nodes[airport_id] = {
                    "label": f"Airport {airport}",
                    "type": "Airport",
                    "details": f"Flights: {metadata.get('flights')}, Avg Delay: {metadata.get('avg_delay')}",
                }

    # Remove duplicates from edges
    unique_edges = []
    seen_edges = set()
    for edge in edges:
        edge_tuple = tuple(sorted((edge["source"], edge["target"])))
        if edge_tuple not in seen_edges:
            unique_edges.append(edge)
            seen_edges.add(edge_tuple)

    return {"nodes": list(nodes.values()), "edges": unique_edges}


def render_knowledge_graph(context_chunks: List[ContextChunk]):
    """
    Renders an interactive knowledge graph visualization using Plotly and NetworkX.
    Extracts graph data from the provided context chunks.
    """
    graph_data = _extract_graph_data_from_context(context_chunks)

    if not graph_data["nodes"]:
        st.info(
            "No discernible graph nodes or relationships found in the retrieved context."
        )
        return

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes to NetworkX graph
    node_labels = {}
    node_types = {}
    node_details = {}
    for node in graph_data["nodes"]:
        node_id = node["label"]  # Use label as ID for NetworkX
        G.add_node(node_id)
        node_labels[node_id] = node["label"]
        node_types[node_id] = node["type"]
        node_details[node_id] = node.get("details", "")

    # Add edges to NetworkX graph
    for edge in graph_data["edges"]:
        source_label = next(
            (n["label"] for n in graph_data["nodes"] if n["label"] == edge["source"]),
            edge["source"],
        )
        target_label = next(
            (n["label"] for n in graph_data["nodes"] if n["label"] == edge["target"]),
            edge["target"],
        )

        # Ensure nodes exist before adding edge
        if source_label in G and target_label in G:
            G.add_edge(source_label, target_label, label=edge["label"])
        else:
            st.warning(
                f"Skipping edge between non-existent nodes: {edge['source']} -> {edge['target']}"
            )

    # Get positions for the nodes
    pos = nx.spring_layout(
        G, k=0.5, iterations=50
    )  # Increased iterations for better spread

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[node_labels[node] for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title=dict(text="Node Connections", side="right"),
                xanchor="left",
            ),
            line_width=2,
        ),
    )

    # Color nodes by degree
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.adj[node])
        node_adjacencies.append(len(adjacencies))
        hover_text = f"<b>{node_labels[node]}</b><br>Type: {node_types[node]}"
        if node_details[node]:
            hover_text += f"<br>Details: {node_details[node]}"
        hover_text += f"<br>Connections: {len(adjacencies)}"
        node_text.append(hover_text)

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="<br>Knowledge Graph Visualization", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Python code for Plotly Graph Visualization",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
