import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import igraph
from igraph import Graph, EdgeSeq


def create_return_hist_plot():
    # Sample data
    data = np.random.normal(loc=0, scale=1, size=1000) + 80

    # Manually calculate histogram data using NumPy
    num_bins = 20  # Example fixed number of bins
    hist_data = np.histogram(data, bins=num_bins, density=True)
    bin_edges = hist_data[1]
    bin_heights = hist_data[0]

    # Create the figure with histogram using the same bin edges
    fig = go.Figure(
        data=[
            go.Histogram(
                x=data,
                histnorm="probability density",
                xbins=dict(
                    start=bin_edges[0],  # Start at the first bin edge
                    end=bin_edges[-1],  # End at the last bin edge
                    size=(bin_edges[1] - bin_edges[0]),  # Bin size based on edges
                ),
                marker=dict(
                    color="gold",
                ),
                opacity=0.5,
            )
        ]
    )

    # Create x and y coordinates for the line plot (outline)
    x_outline = np.repeat(bin_edges, 2)[1:-1]
    y_outline = np.repeat(bin_heights, 2)

    # Add the line plot to trace the histogram's outline
    fig.add_trace(
        go.Scatter(
            x=x_outline,
            y=y_outline,
            mode="lines",
            line=dict(color="gold", width=3),
            name="Outline",
        )
    )

    # Update layout for a polished look with custom dimensions and no padding
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability Density",
        bargap=0,
        template="plotly_white",
        height=250,
        width=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        yaxis=dict(showgrid=False),
    )

    return fig


def create_scenario_tree_plot(target_node=20):
    nr_vertices = 25
    v_label = list(map(str, range(nr_vertices)))

    # Create a tree graph
    G = Graph.Tree(nr_vertices, 2)  # 2 stands for children number
    lay = G.layout("tree")

    # Position the nodes
    position = {k: lay[k] for k in range(nr_vertices)}

    # Calculate new Y positions for top-down layout where children have the same Y
    levels = {}
    for v in range(nr_vertices):
        level = G.shortest_paths_dijkstra(v, 0)[0][
            0
        ]  # Extract the level value from the list
        levels[v] = level

    Y_positions = {k: -levels[k] for k in range(nr_vertices)}  # Top-down structure

    Xn = [position[k][0] for k in range(nr_vertices)]
    Yn = [Y_positions[k] for k in range(nr_vertices)]

    es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [Y_positions[edge[0]], Y_positions[edge[1]], None]

    labels = v_label

    # Determine the path from node 0 to the target_node (if provided)
    path_edges = []
    if target_node is not None and target_node < nr_vertices:
        path = G.get_shortest_paths(0, to=target_node, mode="ALL")[0]
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # PLOT
    fig = go.Figure()

    # Plot edges
    for edge in E:
        if edge in path_edges or (edge[1], edge[0]) in path_edges:
            fig.add_trace(
                go.Scatter(
                    x=[position[edge[0]][0], position[edge[1]][0], None],
                    y=[Y_positions[edge[0]], Y_positions[edge[1]], None],
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hoverinfo="none",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[position[edge[0]][0], position[edge[1]][0], None],
                    y=[Y_positions[edge[0]], Y_positions[edge[1]], None],
                    mode="lines",
                    line=dict(color="rgb(210,210,210)", width=1),
                    hoverinfo="none",
                )
            )

    # Plot nodes
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            name="bla",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color="#000000",  #'#DB4551',
                line=dict(color="rgb(0,0,0)", width=1),
            ),
            text="<b>Internal Energy (kWh): <span style='color: orange;'>50</span></b><br>"
            + "<b>Cumulated Profit ($): <span style='color: orange;'>8</span></b><br>"
            + "<b>24h Future Potential ($): <span style='color: orange;'>123</span></b><br>",
            hoverinfo="text",
            opacity=0.8,
        )
    )

    def make_annotations(pos, text, font_size=10, font_color="rgb(250,250,250)"):
        L = len(pos)
        if len(text) != L:
            raise ValueError("The lists pos and text must have the same len")
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text="<b>" + labels[k],
                    x=pos[k][0],
                    y=Y_positions[k],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                )
            )
        return annotations

    # Example edge labels
    edge_labels = ["1"] * len(E)  # List of labels for each edge
    edge_positions = [
        (
            0.5 * (position[edge[0]][0] + position[edge[1]][0]),
            0.5 * (Y_positions[edge[0]] + Y_positions[edge[1]]),
        )
        for edge in E
    ]

    # Add edge annotations
    def make_edge_annotations(
        edge_positions, edge_labels, font_size=10, font_color="rgb(0,0,0)"
    ):
        annotations = []
        for i, pos in enumerate(edge_positions):
            annotations.append(
                dict(
                    text=edge_labels[i],  # Label for the edge
                    x=pos[0],
                    y=pos[1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                )
            )
        return annotations

    axis = dict(
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    fig.update_layout(
        annotations=make_annotations(position, v_label),
        # + make_edge_annotations(edge_positions, edge_labels),
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def create_scenario_tree_plot_old_but_working():
    nr_vertices = 25
    v_label = list(map(str, range(nr_vertices)))
    # Perform BFS on the graph to get the order of nodes
    G = Graph.Tree(nr_vertices, 2)  # 2 stands for children number
    lay = G.layout("tree")

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    labels = v_label

    # PLOT
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            name="bla",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color="#000000",  #'#DB4551',
                line=dict(color="rgb(0,0,0)", width=1),
            ),
            text="State Info A<br>State Info B<br>State Info C",
            hoverinfo="text",
            opacity=0.8,
        )
    )

    def make_annotations(pos, text, font_size=10, font_color="rgb(250,250,250)"):
        L = len(pos)
        if len(text) != L:
            raise ValueError("The lists pos and text must have the same len")
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text="<b>"
                    + labels[
                        k
                    ],  # or replace labels with a different list for the text within the circle
                    x=pos[k][0],
                    y=2 * M - position[k][1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                )
            )
        return annotations

    # Example edge labels
    edge_labels = ["1"] * len(E)  # List of labels for each edge
    edge_positions = [
        (
            0.5 * (position[edge[0]][0] + position[edge[1]][0]),
            2 * M - 0.5 * (position[edge[0]][1] + position[edge[1]][1]),
        )
        for edge in E
    ]

    # Add edge annotations
    def make_edge_annotations(
        edge_positions, edge_labels, font_size=10, font_color="rgb(0,0,0)"
    ):
        annotations = []
        for i, pos in enumerate(edge_positions):
            annotations.append(
                dict(
                    text=edge_labels[i],  # Label for the edge
                    x=pos[0],
                    y=pos[1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                )
            )
        return annotations

    axis = dict(
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    fig.update_layout(
        annotations=make_annotations(position, v_label)
        + make_edge_annotations(edge_positions, edge_labels),
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def create_dual_violin_plot(
    df, blue_col, yellow_col, sample_df, column_ui, height=350, width=800
):
    fig = go.Figure()

    # Blue side of the violin plot
    fig.add_trace(
        go.Violin(
            x=[0] * len(df[blue_col]),
            y=df[blue_col],
            side="negative",
            line_color="blue",
            points=False,
            name=blue_col,
        )
    )

    # Yellow side of the violin plot
    fig.add_trace(
        go.Violin(
            x=[0] * len(df[yellow_col]),
            y=df[yellow_col],
            side="positive",
            line_color="orange",
            points=False,
            name=yellow_col,
        )
    )

    # Adding the red point for the sample data
    if sample_df is not None and not sample_df.empty:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=sample_df[blue_col],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Sample Blue",
            )
        )

    fig.update_layout(
        violingap=0,
        violinmode="overlay",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title=column_ui,
        ),
        yaxis=dict(showgrid=False),
        height=height,
        width=width,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Ensuring the graph is fully shown in X and Y
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(automargin=True)

    return fig


def create_radar_plot(
    baseline,
    max_values,
    current,
    custom_labels=None,
    width: int = 800,
    height: int = 250,
):
    # Ensure the radar plot is closed by repeating the first value at the end
    categories = list(baseline.keys())
    categories_repeat = categories + [categories[0]]

    baseline_values = list(baseline.values()) + [list(baseline.values())[0]]
    max_values_values = list(max_values.values()) + [list(max_values.values())[0]]
    current_values = list(current.values()) + [list(current.values())[0]]

    # Prepare the figure
    fig = go.Figure()

    # Baseline trace
    fig.add_trace(
        go.Scatterpolar(
            r=baseline_values,
            theta=categories_repeat,
            fill="toself",
            name="Baseline",
            marker=dict(color="grey", opacity=0.6),
            line=dict(color="grey", dash="solid"),
        )
    )

    # Current values trace with custom labels
    fig.add_trace(
        go.Scatterpolar(
            r=current_values,
            theta=categories_repeat,
            text=custom_labels + [custom_labels[0]]
            if custom_labels
            else [""] * len(categories_repeat),
            mode="markers+lines+text",
            textposition="top right",
            fill="toself",
            name="Current",
            marker=dict(color="blue", opacity=1),
            line=dict(color="blue", dash="solid"),
            fillcolor="rgba(0, 0, 255, 0.3)",  # Blue fill with reduced opacity
        )
    )

    # Max values trace
    fig.add_trace(
        go.Scatterpolar(
            r=max_values_values,
            theta=categories_repeat,
            fill="none",
            name="Empirical Best",
            line=dict(color="black", dash="dot"),
        )
    )

    # Update layout
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                showline=False,
                showticklabels=False,
                showgrid=False,
            ),
            angularaxis=dict(
                showline=False,
                showticklabels=True,  # Enabling tick labels for categories
                showgrid=False,
            ),
        ),
        showlegend=True,
        margin=dict(l=40, r=0, t=40, b=40),
    )

    # Define figure size
    fig.update_layout(width=width, height=height)

    return fig


def plotly_marginal_price_plot(resource_data, color_map, current_demand):
    # Prepare lists from the input data
    resources = []
    costs = []
    capacities = []
    for res, entries in resource_data.items():
        for entry in entries:
            resources.append(res)
            costs.append(entry["cost"])
            capacities.append(entry["capacity"])

    # Calculate cumulative capacities for stacking
    cumulative_capacities = [sum(capacities[: i + 1]) for i in range(len(capacities))]
    x_vals = [0] + cumulative_capacities  # Add zero at the beginning for the x-axis

    # Determine the current price based on the current demand
    current_price = next(
        cost for cap, cost in zip(cumulative_capacities, costs) if current_demand <= cap
    )

    # Create the plot
    fig = go.Figure()

    # Add filled areas for each resource
    legend_added = set()
    for i, resource in enumerate(resources):
        x_start = cumulative_capacities[i] - capacities[i] if i > 0 else 0
        x_end = cumulative_capacities[i]

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end, x_end, x_start],
                y=[0, 0, costs[i], costs[i]],
                fill="toself",
                fillcolor=color_map[resource]["fill_color"],
                line=dict(color=color_map[resource]["line_color"]),
                name=resource,
                mode="none",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[costs[i], costs[i]],
                mode="lines",
                line=dict(
                    color=color_map[resource]["line_color"],
                    width=2,
                ),
                name=resource,
                showlegend=resource not in legend_added,
            )
        )

        legend_added.add(resource)

    # Add current demand line
    fig.add_trace(
        go.Scatter(
            x=[current_demand, current_demand],
            y=[0, current_price],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Current Demand",
            showlegend=False,
        )
    )

    # Add dashed y-axis line at the intersection of current demand and price
    fig.add_trace(
        go.Scatter(
            x=[0, current_demand],
            y=[current_price, current_price],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Price at Current Demand",
            showlegend=False,
        )
    )

    # Add annotations for the current price
    fig.add_trace(
        go.Scatter(
            x=[current_demand],
            y=[current_price],
            mode="markers+text",
            # Break text into 2 lines for better readability
            text=[
                f"<b>Spot Price:</b> {current_price} $/MWh<br><b>Marginal Unit:</b> Coal Plant 2"
            ],
            textposition="top right",
            textfont=dict(color="red"),
            marker=dict(color="red", size=10),
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Capacity (MW)",
        yaxis_title="Marginal Cost ($/MW)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False),  # Remove x-axis grid lines
        yaxis=dict(showgrid=False),  # Remove y-axis grid lines
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plotly_unit_commitment_plot(
    nuclear: np.ndarray,
    hydro: np.ndarray,
    solar: np.ndarray,
    wind: np.ndarray,
    thermal_coal: np.ndarray,
    thermal_gas: np.ndarray,
    storage: np.ndarray,
):
    # Generate artificial data
    hours = np.arange(0, 24)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Hour": hours,
            "Nuclear": nuclear,
            "Hydro": hydro,
            "Wind": wind,
            "Thermal (Coal)": thermal_coal,
            "Thermal (Gas)": thermal_gas,
            "Storage": storage,
            "Solar": solar,
        }
    )

    # Melt the DataFrame
    df_melted = df.melt(id_vars="Hour", var_name="Resource", value_name="Generation")

    # Create the stacked area plot
    fig = px.area(
        df_melted,
        x="Hour",
        y="Generation",
        color="Resource",
        color_discrete_map={
            "Nuclear": "violet",
            "Hydro": "blue",
            "Solar": "yellow",
            "Wind": "green",
            "Thermal (Coal)": "black",
            "Thermal (Gas)": "red",
            "Storage": "orange",
        },
    )

    # Update layout for white background and titles
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Power Generation (MW)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def plotly_filled_grad_line_chart(
    df: pd.DataFrame,
    col: str,
    colorscale: list | str = "Greens",
    line_color: str = "green",
    line_width: int = 3,
    show_legend: bool = True,
    width: int = 800,
    height: int = 250,
):
    # Create the area plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            fill="tozeroy",  # Fill to the x-axis
            # mode="none",  # No markers or lines
            mode="lines",
            line=dict(color=line_color, width=line_width),
            fillgradient=dict(
                colorscale=colorscale,  # Gradient color scale
                type="vertical",  # Orientation of the gradient
            ),
        )
    )

    # Update layout to remove background
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        margin=dict(l=0, r=0, t=0, b=60),
        showlegend=show_legend,
    )

    # Define figure size
    fig.update_layout(width=width, height=height)

    return fig


def plotly_colored_line_chart(
    df: pd.DataFrame,
    col: str,
    line_color: str = "green",
    line_width: int = 1.5,
    show_legend: bool = True,
    width: int = 800,
    height: int = 250,
):
    # Create the area plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            line=dict(color=line_color, width=line_width),
        )
    )

    # Update layout to remove background
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        margin=dict(l=0, r=0, t=0, b=60),
        showlegend=show_legend,
    )

    # Define figure size
    fig.update_layout(width=width, height=height)

    return fig


def plotly_line_chart(
    df: pd.DataFrame,
    cols: list[str],
    show_legend: bool = True,
    width: int = 800,
    height: int = 250,
) -> go.Figure:
    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each column
    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    # Update layout to remove background
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        margin=dict(l=0, r=0, t=0, b=60),
        showlegend=show_legend,
    )

    # Define figure size
    fig.update_layout(width=width, height=height)

    return fig


def plotly_hist_plot(
    df: pd.DataFrame,
    cols: list[str],
    show_legend: bool = True,
    width: int = 800,
    height: int = 250,
) -> go.Figure:
    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each column
    for col in cols:
        fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=100))

    # Update layout to remove background
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=0, b=60),
        showlegend=show_legend,
    )

    # Define figure size
    fig.update_layout(width=width, height=height)

    return fig


def create_tou_heatmap(
    default_value: float, rate_changes: list[dict] = []
) -> go.Figure:
    """
    Creates a heatmap for ToU rates with specified changes.

    Parameters:
    - default_value (float): The default rate for all times.
    - rate_changes (list of dicts): Each dictionary should have keys "start", "end", "days", and "value".
      "start" and "end" are integers representing hours (0-23).
      "days" is a list of integers representing days (0=Monday, 1=Tuesday, ..., 6=Sunday).
      "value" is the new rate to be applied for the specified period.

    Returns:
    - A Plotly figure object.
    """

    custom_colorscale = [
        [0, "rgb(255, 238, 205)"],  # Yellow
        [0.5, "rgb(255, 204, 33)"],  # Orange
        [1, "rgb(255, 51, 51)"],  # Red
    ]

    # colorscale = "reds" if rate_changes else "Greys"
    colorscale = custom_colorscale if rate_changes else "Greys"
    show_colorbar = True if rate_changes else False

    # Initialize the time-of-use grid
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [f"{h:02}:00" for h in range(24)]
    tou_grid = np.full((7, 24), default_value)

    # Apply rate changes
    for change in rate_changes:
        start_idx = change["start"]
        end_idx = change["end"]
        for day_idx in change["days"]:
            tou_grid[day_idx, start_idx:end_idx] = change["value"]

    # Separate data for weekdays and weekends
    weekdays_grid = tou_grid[:5][::-1]  # Monday to Friday, reversed
    weekends_grid = tou_grid[5:][::-1]  # Saturday and Sunday, reversed
    weekday_labels = days[:5][::-1]
    weekend_labels = days[5:][::-1]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[5, 2],  # Larger height for weekdays, smaller for weekends
        shared_xaxes=True,
        vertical_spacing=0.1,  # Spacing between the subplots
    )

    # Weekdays heatmap
    fig.add_trace(
        go.Heatmap(
            z=weekdays_grid,
            x=hours,
            y=weekday_labels,
            colorscale=colorscale,
            zmin=np.min(tou_grid),
            zmax=np.max(tou_grid),
            showscale=show_colorbar,
            colorbar=dict(
                title="Rate ($/kWh)",
                outlinewidth=0,
            ),
            xgap=1,  # Gap between hours
            ygap=1,  # Gap between days
        ),
        row=1,
        col=1,
    )

    # Weekends heatmap
    fig.add_trace(
        go.Heatmap(
            z=weekends_grid,
            x=hours,
            y=weekend_labels,
            colorscale=colorscale,
            zmin=np.min(tou_grid),
            zmax=np.max(tou_grid),
            showscale=False,  # No separate color scale for weekends
            xgap=1,  # Gap between hours
            ygap=1,  # Gap between days
        ),
        row=2,
        col=1,
    )

    # Customize layout
    fig.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            tickvals=list(range(24)),
            ticktext=[f"{h:02}:00" for h in range(24)],
            tickmode="array",
            automargin=True,
            side="top",
            tickangle=0,
            dtick=1,
            gridcolor="lightgrey",
            mirror=True,
            # showline=True,
            # linecolor='black',
            # linewidth=1,
        ),
        xaxis2=dict(
            tickvals=list(range(24)),
            ticktext=[f"{h:02}" for h in range(24)],
            tickmode="array",
            automargin=True,
            side="bottom",
            tickangle=0,
            dtick=1,
            gridcolor="lightgrey",
            mirror=True,
            # showline=True,
            # linecolor='black',
            # linewidth=1,
        ),
        yaxis=dict(
            tickvals=list(range(5)),
            ticktext=weekday_labels,
            tickmode="array",
            automargin=True,
        ),
        yaxis2=dict(
            tickvals=list(range(2)),
            ticktext=weekend_labels,
            tickmode="array",
            automargin=True,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # Modify figure height
    fig.update_layout(height=260)

    return fig


def plot_temperature_power(
    df,
    temp_column,
    baseline_temp_column,
    heating_setpoint_column,
    cooling_setpoint_column,
    power_column,
    baseline_power_column,
):
    # Create subplots with two rows
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Temperature Variation", "Power Consumption")
    )

    # Temperature Variation Plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[temp_column],
            name="AI Agent Temperature",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[baseline_temp_column],
            name="Baseline Temperature",
            line=dict(color="grey"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[heating_setpoint_column],
            name="Heating Setpoint",
            fill=None,
            mode="none",
            line=dict(color="blue", width=0),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[cooling_setpoint_column],
            name="Cooling Setpoint",
            fill="tonexty",
            mode="none",
            line=dict(color="red", width=0),
        ),
        row=1,
        col=1,
    )

    # Power Consumption Plot
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[power_column],
            name="AI Agent Power",
            marker=dict(color=df[power_column], coloraxis="coloraxis"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[baseline_power_column],
            name="Baseline Power",
            marker=dict(color=df[baseline_power_column], coloraxis="coloraxis"),
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        coloraxis=dict(colorscale="YlOrRd"), showlegend=False, plot_bgcolor="white"
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Power Consumption (kW)", row=2, col=1)

    return fig


def plot_q_factors_as_bars(q_factors, title: str = "", fixed_min=0, fixed_max=1):
    """
    Plot a bar graph of Q-factors using Plotly with improved aesthetics.

    Parameters:
    q_factors (list): A list of Q-factor values.
    fixed_min (float): Fixed minimum value for the y-axis.
    fixed_max (float): Fixed maximum value for the y-axis.
    """
    # Assigning colors based on the value and intensity of Q-factors
    max_q = max(q_factors)
    min_q = min(q_factors)

    colors = [
        f"rgba(0, 128, 34, {0.5 + 0.5 * (q / max_q)})"
        if q > 0
        else f"rgba(140, 3, 3, {0.5 + 0.5 * (abs(q) / abs(min_q))})"
        for q in q_factors
    ]

    # Creating the bar graph
    fig = go.Figure(
        data=[go.Bar(x=list(range(len(q_factors))), y=q_factors, marker_color=colors)]
    )

    # Updating layout for a cleaner white background
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(t=40, b=0, l=0, r=0),  # Adjust top margin to make space for title,
        height=200,
    )
    fig.update_yaxes(
        showgrid=False,
        range=[fixed_min, fixed_max]
        if fixed_min is not None and fixed_max is not None
        else None,
    )

    return fig


def plot_q_factors_as_circles(
    q_factors,
    min_range=-1,
    max_range=1,
    titles=[
        "<b>Action 1</b> (Sell)",
        "<b>Action 2</b> (Do Nothing)",
        "<b>Action 3</b> (Buy)",
        "<b>Action 4",
        "<b>Action 5",
    ],
):
    if len(q_factors) not in [3, 5]:
        raise ValueError("The number of Q-factors should be 3 or 5.")

    titles = titles[: len(q_factors)]
    fig = make_subplots(rows=1, cols=len(q_factors))

    # Normalize Q-factors
    q_factors = [round(q / sum(np.abs(q_factors)), 2) for q in q_factors]
    # q_factors = np.round(np.tanh(np.array(q_factors) / 10000), 1)

    for idx, val in enumerate(q_factors, 1):
        # Color based on value
        color = "green" if val > 0 else "red"

        # Circle dimension proportional to absolute value
        size = abs(val)

        # Circle trace
        circle = {
            "type": "scatter",
            "x": [0],
            "y": [0],
            "mode": "markers+text",
            "text": [str(val * 100) + "%"],
            "textposition": "top center",
            "textfont": {"size": 18, "color": "black"},
            "marker": {
                "size": [
                    size * 150
                ],  # Multiply by a constant to make the circle visible
                "color": color,
            },
            "hoverinfo": "none",
        }

        fig.add_trace(circle, row=1, col=idx)

        # Adjust the axis range
        fig.update_xaxes(range=[min_range, max_range], row=1, col=idx)
        fig.update_yaxes(range=[min_range / 4, max_range / 6], row=1, col=idx)

        fig.update_xaxes(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=1,
            col=idx,
        )
        fig.update_yaxes(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=1,
            col=idx,
        )
        fig.update_layout(
            showlegend=False,
        )

    # Adding control titles under each circle
    for i, title in enumerate(titles, 1):
        fig.add_annotation(
            dict(
                font=dict(color="black", size=15, family="Arial, bold"),
                x=0,
                y=-0.2,
                showarrow=False,
                text=title,
                xref="x" + str(i),
                yref="y" + str(i),
                xanchor="center",
                yanchor="bottom",
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=0, b=0, l=0, r=0),
        height=150,
    )

    return fig


def plotly_sankey_plot():
    # Define the labels for the nodes
    labels = labels = [
        "Total Expenses",
        "Fixed Charges",
        "Reactive<br>Power",
        "Power",
        "Energy",
        "Demand Charge",
        "On-Peak",
        "Mid-Peak",
        "Off-Peak",
        "T&D",
        "Power Factor",
    ]

    # Define the source and target nodes for the links
    source = [0, 0, 0, 0, 3, 4, 4, 4, 4, 2]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Define the values (weights) of the links
    values = [1000, 1000, 3000, 7000, 3000, 2500, 2000, 1000, 1500, 1000]

    # Define the node colors (all shades of green)
    node_colors = node_colors = [
        "#006400",
        "#008000",
        "#008000",
        "#008000",
        "#008000",
        "#228B22",
        "#228B22",
        "#228B22",
        "#228B22",
        "#228B22",
        "#228B22",
    ]

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    color=[
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                        "#c8e8cf",
                    ],
                ),
                textfont=dict(color="black", size=12),
            )
        ]
    )

    # Update layout to have a white background
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=60),
        width=800,
        height=250,
    )

    return fig
