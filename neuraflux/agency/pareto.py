import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class ParetoFrontier:
    r"""Pareto Frontier class, with multi-objective optimization
    functionalities.

    Parameters
    ----------
    points : numpy.ndarray
        Numpy array of dimension (n_points,n_objectives).
    op     : str
        Desired operation, use 'min' or 'max'.

    Returns
    -------
    ParetoFrontier object
    """

    # ---------------------------#
    # - INITIALIZATION FUNCTION -#
    # ---------------------------#
    # @ Ysael Desage
    # Function initializing the class, its inherent properties and variables.
    def __init__(self, points, op="min"):
        # Initialization
        self.points = points  # Origin. points
        self.pareto_mask = is_pareto_efficient(points, op=op)  # Pareto mask
        self.pareto_points = points[self.pareto_mask]  # Pareto points
        self.op = op  # Operation
        # -------------------#
        # - END OF FUNCTION -#
        # -------------------#

    # ---------------------------#
    # - INITIALIZATION FUNCTION -#
    # ---------------------------#
    # @ Ysael Desage
    # Function initializing the class, its inherent properties and variables.
    def initial_position(self, method: str = "max", objective_index: int = 0):
        # Find the index of the point that maximizes the desired objective
        if method == "max":
            pareto_idx = np.argmax(self.pareto_points[:, objective_index])

        # Find the index of the point that minimizes the desired objective
        elif method == "min":
            pareto_idx = np.argmin(self.pareto_points[:, objective_index])

        # Farthest norm in +/- direction based on desired operation
        elif method == "norm":
            # Define starting point - Farthest in positive or negative direction
            custom_norm = np.sum(
                np.sign(self.pareto_points) * np.square(self.pareto_points),
                axis=1,
            )
            pareto_idx = (
                np.argmin(custom_norm) if self.op == "min" else np.argmax(custom_norm)
            )  # Pareto mask index
        else:
            raise ValueError("Method should be 'max', 'min' or 'norm'.")

        self.point = self.pareto_points[pareto_idx]  # Point
        self.idx = np.where(np.all(self.points == self.point, axis=1))[0][0]

        # Function return | Initial point index
        return self.idx
        # -------------------#
        # - END OF FUNCTION -#
        # -------------------#

    def get_current_index(self):
        """
        Get the index of the current point in the original points array.
        """
        current_idx = np.where(np.all(self.points == self.point, axis=1))[0][0]
        return current_idx

    def move_to_next_closest(self):
        """
        Move to the next closest Pareto-optimal point based on the first objective.
        If there is no more, return None.
        """
        if self.pareto_points.shape[0] <= 1:
            return None

        # Exclude the current point from the set of points to consider
        # Find the corresponding index in self.pareto_points
        pareto_idx = np.where(np.all(self.pareto_points == self.point, axis=1))[0][0]
        other_pareto_points = np.delete(self.pareto_points, pareto_idx, axis=0)

        # Find the closest point in terms of the first objective
        closest_idx = closest_node(
            self.point, other_pareto_points, objectives={0: "floor"}
        )

        if closest_idx is None:
            return None

        # Update the current point
        self.point = other_pareto_points[closest_idx]

        # Update self.idx to match the new point in the original points array
        self.idx = np.where(np.all(self.points == self.point, axis=1))[0][0]

        return self.idx

    def delta_to_next_closest(self):
        """
        Anticipate the delta between the current value of the first objective function
        and the new one at the closest Pareto-optimal point.
        Return the delta as a float.
        """
        if self.pareto_points.shape[0] <= 1:
            return None

        # Exclude the current point from the set of points to consider
        pareto_idx = np.where(np.all(self.pareto_points == self.point, axis=1))[0][0]
        other_pareto_points = np.delete(self.pareto_points, pareto_idx, axis=0)

        # Find the closest point in terms of the first objective
        closest_idx = closest_node(
            self.point, other_pareto_points, objectives={0: "floor"}
        )

        if closest_idx is None:
            return None

        # Calculate the delta for the first objective
        next_point = other_pareto_points[closest_idx]
        delta = next_point[0] - self.point[0]

        return delta


################################################################################
### 4. PARETO FUNCTIONS
################################################################################


# ------------------------------------------#
# - MAIN PARETO FRONT CALCULATOR FUNCTION  -#
# ------------------------------------------#
# @ Ysael Desage
def is_pareto_efficient(objectives, op="min"):
    r"""Function returning whether each point with associated objectives is
    Pareto efficient. Fastest implementation.

    Parameters
    ----------
    objectives : numpy.ndarray
        Numpy array of dimension (n_points,n_objectives)
    op : str
        Desired operation. Choose between 'min' or 'max'.

    Returns
    -------
    is_efficient : numpy.ndarray
        Numpy boolean array of dim (n_points,), indicating if each point is
        Pareto efficient or not.
    """

    # Initialization
    is_efficient = np.arange(objectives.shape[0])
    n_points = objectives.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for

    # Convert to maximization problem, if required
    if op == "max":
        objectives = -1 * objectives

    # Main loop
    while next_point_index < len(objectives):
        nondominated_point_mask = np.any(
            objectives < objectives[next_point_index], axis=1
        )
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        objectives = objectives[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    # Create boolean array from is efficient indices
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True

    # Function return | Pareto-eficient boolean array
    return is_efficient_mask
    # -------------------#
    # - END OF FUNCTION -#
    # -------------------#


# --------------------------------------------#
# - PARETO EFFICIENCY CALCULATION FUNCTION 1 -#
# --------------------------------------------#
# @ Ysael Desage
def is_pareto_efficient_1(objectives, op="min"):
    r"""Function returning whether each point with associated objectives is
    Pareto efficient. Slow for many datapoints, fastest for many objectives.

    Parameters
    ----------
    objectives : numpy.ndarray
        Numpy array of dimension (n_points,n_objectives)
    op : str
        Desired operation. Choose between 'min' or 'max'.

    Returns
    -------
    is_efficient : numpy.ndarray
        Numpy boolean array of dim (n_points,), indicating if each point is
        Pareto efficient or not.
    """

    # Initialization
    is_efficient = np.ones(objectives.shape[0], dtype=bool)

    # Convert to maximization problem, if required
    if op == "max":
        objectives = -1 * objectives

    # Loop over points
    for i, c in enumerate(objectives):
        is_efficient[i] = np.all(np.any(objectives[:i] > c, axis=1)) and np.all(
            np.any(objectives[i + 1 :] > c, axis=1)
        )

    # Function return | Pareto-eficient boolean array
    return is_efficient
    # -------------------#
    # - END OF FUNCTION -#
    # -------------------#


# --------------------------------------------#
# - PARETO EFFICIENCY CALCULATION FUNCTION 2 -#
# --------------------------------------------#
# @ Ysael Desage
def is_pareto_efficient_2(objectives, op="min"):
    r"""Function returning whether each point with associated objectives is
    Pareto efficient. Fast for many datapoints, less for many objectives.

    Parameters
    ----------
    objectives : numpy.ndarray
        Numpy array of dimension (n_points,n_objectives)
    op : str
        Desired operation. Choose between 'min' or 'max'.

    Returns
    -------
    is_efficient : numpy.ndarray
        Numpy boolean array of dim (n_points,), indicating if each point is
        Pareto efficient or not.
    """

    # Initialization
    is_efficient = np.ones(objectives.shape[0], dtype=bool)

    # Convert to maximization problem, if required
    if op == "max":
        objectives = -1 * objectives

    # Loop over points
    for i, c in enumerate(objectives):
        if is_efficient[i]:
            # Keep any point with a lower cost (1), and keep self (2)
            is_efficient[is_efficient] = np.any(objectives[is_efficient] < c, axis=1)
            is_efficient[i] = True

    # Function return | Pareto-eficient boolean array
    return is_efficient
    # -------------------#
    # - END OF FUNCTION -#
    # -------------------#


################################################################################
### 5. NDIM GEOMETRY FUNCTIONS
################################################################################


# --------------------------------------#
# - CLOSEST NODE CALCULATION FUNCTION  -#
# --------------------------------------#
# @ Ysael Desage
def closest_node(point, points, distances=False, objectives={}):
    r"""Function returning the closest point in a set, to an input point.
    Optimal implementation.

    Parameters
    ----------
    point : numpy.ndarray
        Point in vector format.
    points : numpy.ndarray
        Set of points to find the closest point from.
    distances : bool
        Whether or not to also return the distances between each point of the
        set.
    objectives : dict
        Dictionary of objective functions to apply a floor/ceiling on.

    Returns
    -------
    closest or closest,dist2 : numpy.ndarray or tuple
        Numpy boolean array of the closest point
    """

    # Initialization
    points = np.asarray(points)  # Make sure set is an 2D array
    deltas = points - point  # Difference vector

    # Dot product with einsum
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)

    if objectives:
        mask = np.zeros(dist_2.shape, dtype=bool)
        for key, val in objectives.items():
            if val == "floor":
                mask = mask | (deltas[:, key] <= 0)
            elif val == "ceiling":
                mask = mask | (deltas[:, key] >= 0)

        if np.all(mask):  # If all points are masked out, return None
            return None

        dist_2 = np.ma.masked_array(dist_2, mask=mask)

    closest = np.argmin(dist_2)  # Closest point

    output = (closest, np.sqrt(dist_2)) if distances else closest

    # Function return | Closest point from inputed set
    return output
    # -------------------#
    # - END OF FUNCTION -#
    # -------------------#


# --------------------------------------#
# - CLOSEST NODE CALCULATION FUNCTION  -#
# --------------------------------------#
# @ Ysael Desage
def closest_node_1(point, points, distances=False):
    r"""Function returning the closest point in a set, to an input point.
    Slower implementation.

    Parameters
    ----------
    point : numpy.ndarray
        Point in vector format.
    points : numpy.ndarray
        Set of points to find the closest point from.
    distances : bool
        Whether or not to also return the distances between each point of the
        set.

    Returns
    -------
    closest or closest,dist2 : numpy.ndarray or tuple
        Numpy boolean array of the closest point
    """

    # Initialization
    points = np.asarray(points)

    # Squared sum calculation
    dist_2 = np.sum((points - point) ** 2, axis=1)
    closest = np.argmin(dist_2)  # Closest point

    output = (closest, np.sqrt(dist_2)) if distances else closest

    # Function return | Closest point from inputed set
    return output
    # -------------------#
    # - END OF FUNCTION -#
    # -------------------#


# Visualization Tools
def plot_pareto_front(points, pareto_points):
    """
    Plot the Pareto front in a 2D space, removing top and right axes and adding arrows.
    """
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c="lightgray", label="All Points")
    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        c="red",
        s=60,
        label="Pareto Front",
    )

    # Customize axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")

    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.legend()
    plt.show()


def plot_pareto_front_plotly(
    sub_df,
    start_idx,
    objective_1="Thermal Comfort",
    objective_2="Energy Consumption",
):
    """
    Plot the Pareto front in a 2D space using Plotly, with updated styles.
    """

    # Create a subplot figure with 1 row and 3 columns with shared axes
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"Controller {c+1}" for c in range(3)],
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for c in range(3):
        # Extract data for each configuration
        Q1 = [sub_df.loc[start_idx, f"Q1_C{c+1}_U{i}"] for i in range(1, 6)]
        Q2 = [sub_df.loc[start_idx, f"Q2_C{c+1}_U{i}"] for i in range(1, 6)]
        points = np.array(list(zip(Q1, Q2)))
        pareto = ParetoFrontier(points, op="max")
        pareto_points = pareto.pareto_points

        # Create traces for all points and Pareto front points
        trace_all = go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker=dict(color="gray", size=10),
            showlegend=False,
        )
        trace_pareto = go.Scatter(
            x=pareto_points[:, 0],
            y=pareto_points[:, 1],
            mode="markers",
            marker=dict(color="red", size=12),
            showlegend=False,
        )

        # Add traces to the subplot
        fig.add_trace(trace_all, row=1, col=c + 1)
        fig.add_trace(trace_pareto, row=1, col=c + 1)

    # Update layout
    fig.update_layout(plot_bgcolor="white", showlegend=False, height=350, width=900)

    # Remove gridlines and zero lines, and set axis titles for the first subplot only
    fig.update_xaxes(
        showgrid=False, zeroline=True, title_text=objective_1 if c == 0 else None
    )
    fig.update_yaxes(
        showgrid=False, zeroline=True, title_text=objective_2 if c == 0 else None
    )
    fig.update_xaxes(title_text="Thermal Comfort", row=1, col=2)
    fig.update_yaxes(title_text="Energy Consumption", row=1, col=1)

    return fig


# Example of how to use the function:
# points = np.random.rand(100, 2) * 100  # Random example points
# pareto_points = points[points[:, 0] + points[:, 1] < 50]  # Example Pareto front points
# plot_pareto_front_plotly(points, pareto_points)


# Visualization Tools
def plot_pareto_front_3d(points, pareto_points):
    """
    Plot the Pareto front in a 2D space with updated styles.
    """
    fig, ax = plt.subplots()
    ax.scatter(
        points[:, 0], points[:, 1], c="lightgray", s=200, label="Controls", zorder=1
    )

    # Sort Pareto points by the first objective (can change to [:, 1] for the second objective)
    sorted_pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    # Connect Pareto points with lines
    ax.plot(
        sorted_pareto_points[:, 0],
        sorted_pareto_points[:, 1],
        color="red",
        linestyle="--",
        zorder=1,
        alpha=1,
    )

    # Plot Pareto front points with lightgrey fill and red border
    ax.scatter(
        sorted_pareto_points[:, 0],
        sorted_pareto_points[:, 1],
        facecolors="lightgray",
        edgecolors="red",
        s=200,
        label="Pareto-Optimal Controls",
        linewidth=3,
        zorder=2,
    )

    # Customize axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")

    ax.set_xlabel("GHG Emissions (gCO2)")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    plt.axis([0, 300, 0, 2000])
    plt.show()

    ax.set_title("3D Pareto Front Visualization")
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.legend()

    plt.show()


if __name__ == "__main_":
    # Example Usage
    np.random.seed(0)
    example_points = np.array(
        [
            [200, 1500],
            [200, 1400],
            [220, 1750],
            [180, 1600],
            [150, 1800],
            [200, 1000],
            [250, 750],
        ]
    )
    pareto = ParetoFrontier(example_points, op="max")
    pareto_points = pareto.pareto_points
    plot_pareto_front(example_points, pareto_points)
    exit()
    # Example Usage for 3D
    np.random.seed(0)
    example_points_3D = np.random.rand(100, 3)
    pareto_3D = ParetoFrontier(example_points_3D, op="max")
    pareto_points_3D = pareto_3D.pareto_points
    plot_pareto_front_3d(example_points_3D, pareto_points_3D)
