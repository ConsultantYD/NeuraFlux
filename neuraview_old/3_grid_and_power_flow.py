import streamlit as st
from sidebar_utils import generate_sidebar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dill
import os
from plot_utils import plotly_unit_commitment_plot, plotly_marginal_price_plot
import pypsa

# Page title and sidebar initialization
st.header("⚡️ Energy Grid Dynamics")
generate_sidebar()

if not os.path.isdir("temp_cache"):
    os.mkdir("temp_cache")

# Import the network
network = pypsa.examples.scigrid_de(from_master=True)

_, title_col1, title_col2 = st.columns((20, 78, 42))
col11, col12 = st.columns(2)

# --------------------------------------------------------------------------------------------------
# LOAD PLOTTING
# --------------------------------------------------------------------------------------------------
# Plot load distribution
load_distribution = (
    network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
)
fig_dict = network.iplot(
    iplot=False,
    bus_sizes=0.05 * load_distribution.copy(),
    bus_colors="blue",
)
# Update layout
load_fig = go.Figure(**fig_dict)
load_fig.update_layout(
    autosize=False,
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),
    # Hide grid lines
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False, showticklabels=False),
)

title_col2.write("#### Load")
col12.plotly_chart(load_fig, use_container_width=True)


# --------------------------------------------------------------------------------------------------
# GENERATION PLOTTING
# --------------------------------------------------------------------------------------------------
# Generators information
gen_distribution = network.generators.groupby(network.generators.bus).sum()["p_nom"]

fig_dict = network.iplot(
    iplot=False, bus_sizes=0.01 * gen_distribution.copy(), bus_colors="red"
)

gen_fig = go.Figure(**fig_dict)
gen_fig.update_layout(
    autosize=False,
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),
    # Hide grid lines
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False, showticklabels=False),
)

title_col1.write("#### Generation")
col11.plotly_chart(gen_fig, use_container_width=True)


# --------------------------------------------------------------------------------------------------
# OPF PLOTTING
# --------------------------------------------------------------------------------------------------
_, second_title_col1, second_title_col2 = st.columns((18, 58, 63))
col21, col22 = st.columns(2)

# Check if cache is empty
if os.path.exists("temp_cache/network.nc"):
    network.import_from_netcdf("temp_cache/network.nc")
else:
    # Power Flow on first day of 2011
    contingency_factor = 0.7
    network.lines.s_max_pu = contingency_factor

    network.lines.loc[["316", "527", "602"], "s_nom"] = 1715

    group_size = 4
    network.storage_units.state_of_charge_initial = 0.0

    for i in range(int(24 / group_size)):
        # set the initial state of charge based on previous round
        if i:
            network.storage_units.state_of_charge_initial = (
                network.storage_units_t.state_of_charge.loc[
                    network.snapshots[group_size * i - 1]
                ]
            )
        network.optimize(
            network.snapshots[group_size * i : group_size * i + group_size],
        )

    network.export_to_netcdf("temp_cache/network.nc")

now = network.snapshots[4]

loading = network.lines_t.p0.loc[now] / network.lines.s_nom

fig, ax = plt.subplots(figsize=(9, 9))
network.plot(
    ax=ax,
    line_colors=abs(loading),
    # line_widths=3 * abs(loading),
    line_widths=3,
    line_cmap=plt.cm.jet,
    bus_sizes=1e-3,
    bus_alpha=0.7,
)
fig.tight_layout()

second_title_col1.write("#### Line Loading")
col21.pyplot(fig, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 8))

plt.hexbin(
    network.buses.x,
    network.buses.y,
    gridsize=20,
    C=network.buses_t.marginal_price.loc[now],
    cmap=plt.cm.jet,
    zorder=3,
)

network.plot(ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0)

# cb = plt.colorbar(location="right", label="LMP ($/MWh)")
fig.tight_layout()

# Make colorbar smaller on vertical axis

second_title_col2.write("#### Locational Marginal Prices")
col22.pyplot(fig, use_container_width=True)

# --------------------------------------------------------------------------------------------------
# GENERATION BREAKDOWN
# --------------------------------------------------------------------------------------------------
st.write("#### Resources Breakdown")
st.write("###### Generation Mix")
icons_dir = "neuraview/media/grid_icons/"
gen_col1, gen_col2, gen_col3, gen_col4, gen_col5, gen_col6 = st.columns(6)
(
    _,
    prog_col1,
    _,
    prog_col2,
    _,
    prog_col3,
    _,
    prog_col4,
    _,
    prog_col5,
    _,
    prog_col6,
    _,
) = st.columns((1, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 1))
gen_col1.image(icons_dir + "gen_nuclear.png", use_column_width=True)
prog_col1.progress(0.2)
gen_col2.image(icons_dir + "gen_hydro.png", use_column_width=True)
prog_col2.progress(0.3)
gen_col3.image(icons_dir + "gen_wind.png", use_column_width=True)
prog_col3.progress(0.8)
gen_col4.image(icons_dir + "gen_solar.png", use_column_width=True)
prog_col4.progress(0.4)
gen_col5.image(icons_dir + "gen_thermal.png", use_column_width=True)
prog_col5.progress(0.3)
gen_col6.image(icons_dir + "storage.png", use_column_width=True)
prog_col6.progress(0.15)

st.write("###### Load Mix")
load_col1, load_col2, load_col3, load_col4, _, _ = st.columns(6)
(
    _,
    prog_col11,
    _,
    prog_col12,
    _,
    prog_col13,
    _,
    prog_col14,
    _,
    _,
    _,
    _,
    _,
) = st.columns((1, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 1))
load_col1.image(icons_dir + "industrial.png", use_column_width=True)
prog_col11.progress(0.7)
load_col2.image(icons_dir + "buildings.png", use_column_width=True)
prog_col12.progress(0.6)
load_col3.image(icons_dir + "ev.png", use_column_width=True)
prog_col13.progress(0.2)
load_col4.image(icons_dir + "storage.png", use_column_width=True)
prog_col14.progress(0.5)


# --------------------------------------------------------------------------------------------------
# UNIT COMMITMENT
# --------------------------------------------------------------------------------------------------
st.divider()
st.write("#### Unit Commitment")

p_by_carrier = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()
p_by_carrier.drop(
    (p_by_carrier.max()[p_by_carrier.max() < 1700.0]).index, axis=1, inplace=True
)

# reorder
cols = [
    "Nuclear",
    "Run of River",
    "Brown Coal",
    "Hard Coal",
    "Gas",
    "Wind Offshore",
    "Wind Onshore",
    "Solar",
]
p_by_carrier = p_by_carrier[cols]
storage = p_by_carrier["Run of River"] - p_by_carrier["Solar"]
storage[0:12] = 0
storage = storage.clip(lower=0)
fig = plotly_unit_commitment_plot(
    nuclear=p_by_carrier["Nuclear"].values,
    hydro=p_by_carrier["Run of River"].values,
    solar=p_by_carrier["Solar"].values,
    wind=p_by_carrier["Wind Offshore"].values + p_by_carrier["Wind Onshore"].values,
    thermal_coal=p_by_carrier["Brown Coal"].values + p_by_carrier["Hard Coal"].values,
    thermal_gas=p_by_carrier["Gas"].values,
    storage=storage.values,
)
st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------------------
# MARGINAL PRICE
# --------------------------------------------------------------------------------------------------
resource_data = {
    "Renewables": [{"cost": 10, "capacity": 200}],
    "Nuclear": [{"cost": 20, "capacity": 100}],
    "Thermal (Coal)": [
        {"cost": 60, "capacity": 80},
        {"cost": 80, "capacity": 100},
        {"cost": 100, "capacity": 50},
    ],
    "Thermal (Gas)": [{"cost": 120, "capacity": 100}, {"cost": 140, "capacity": 50}],
}

color_map = {
    "Renewables": {"fill_color": "rgba(0,128,0,0.1)", "line_color": "#008000"},
    "Nuclear": {"fill_color": "rgba(138,43,226,0.1)", "line_color": "#8A2BE2"},
    "Thermal (Coal)": {"fill_color": "rgba(0,0,0,0.1)", "line_color": "#000000"},
    "Thermal (Gas)": {"fill_color": "rgba(255,99,71,0.1)", "line_color": "#FF6347"},
    # "Oil": {"fill_color": "rgba(139,69,19,0.1)", "line_color": "#8B4513"},
}

current_demand = 400

st.write("#### Marginal Price")
fig = plotly_marginal_price_plot(resource_data, color_map, current_demand)
st.plotly_chart(fig, use_container_width=True)
