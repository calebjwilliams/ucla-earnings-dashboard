#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# In[8]:


# Load data
grad_numbers_data = pd.read_csv("Graduates_by_CIP_and_institution_2022.csv")
earnings_data = pd.read_csv("earnings_and_enrollment_by_institution.csv")

# Define institution groups
UCLA_UNITID = 110662
non_UCLA_UCs = [110635, 110680, 110644, 110653, 110705, 110714, 110671]
cal_states = [110486, 110495, 110510, 110529, 110538, 110547, 110556, 110565, 110574,
              110583, 110592, 110608, 110617, 111188, 115755]

# Tag schools
earnings_data["UCLA"] = earnings_data["unitid"] == UCLA_UNITID
earnings_data["non_UCLA_UC"] = earnings_data["unitid"].isin(non_UCLA_UCs)
earnings_data["CSU"] = earnings_data["unitid"].isin(cal_states)

# Filter and prep
enrollment = earnings_data[
    (earnings_data["total_enrollment"] > 0) &
    (earnings_data["6_yr_median_earnings"] > 0) &
    (earnings_data["10_yr_median_earnings"] > 0)
].copy()
enrollment["log_total_enrollment"] = np.log(enrollment["total_enrollment"])

enrollment["school_group"] = np.select(
    [
        enrollment["UCLA"],
        enrollment["non_UCLA_UC"],
        enrollment["CSU"]
    ],
    [
        "UCLA",
        "Other UCs",
        "CSU"
    ],
    default="Other Schools"
)

# Grad rates
grad_rate_by_school = grad_numbers_data.groupby("unitid")["graduation_rate"].mean().reset_index()
enrollment = enrollment.merge(grad_rate_by_school, on="unitid", how="left").dropna(subset=["graduation_rate"])

# Schools we want to compare UCLA with (from clustering) 
comparison_unitids = [
    110404, 123961, 130794, 131469, 131496, 135726, 139658, 144050, 147767, 152080,
    160755, 162928, 164924, 164988, 165015, 166027, 166683, 167358, 168148, 179867,
    182670, 186131, 190150, 190415, 193900, 194824, 195030, 196413, 198419, 199847,
    201645, 211440, 215062, 217156, 221999, 227757, 243744
]
valid_schools = grad_numbers_data[grad_numbers_data["unitid"].isin(comparison_unitids)]
school_options = [{'label': name, 'value': name} for name in sorted(valid_schools["inst_name"].unique()) if name != "University of California-Los Angeles"]




# In[9]:


import pandas as pd
import glob
import re

# Read all CSV files in directory
files = sorted(glob.glob("*.csv"))

grad_rate_dfs = []

for file in files:
    df = pd.read_csv(file)
    grad_col = [col for col in df.columns if re.search(r"^(DFR|DRVGR)\d{4}_RV.*Graduation rate.*total cohort", col)]
    if len(grad_col) == 1:
        year = int(re.search(r"\d{4}", grad_col[0]).group())
        df = df.rename(columns={grad_col[0]: "gradrate"})
        df["year"] = year
        grad_rate_dfs.append(df[["unitid", "institution name", "gradrate", "year"]])

grad_rates = pd.concat(grad_rate_dfs, ignore_index=True)
grad_rates["gradrate"] = grad_rates["gradrate"] / 100  # convert % to proportion


# In[10]:


# CIP codes of interest and schools of interest
cip_codes = [11, 14, 42, 26, 45]
unitids = comparison_unitids + [UCLA_UNITID]

# Filter 2022 data
grads_filtered = grad_numbers_data[
    (grad_numbers_data["gen_cip_code"].isin(cip_codes)) &
    (grad_numbers_data["unitid"].isin(unitids))
].copy()

# Calculate share of field per school
grads_filtered["share"] = grads_filtered["expected_graduates"] / grads_filtered["total_enrollment"]

# Keep relevant columns
school_cips = grads_filtered[["unitid", "inst_name", "gen_cip_code", "general_field", "share", "total_enrollment"]].drop_duplicates()


# In[11]:


# Merge historical grad rate with CIP + enrollment data
merged = pd.merge(
    grad_rates,
    school_cips,
    on="unitid",
    how="inner"
)

merged["expected_graduates"] = merged["gradrate"] * merged["share"] * merged["total_enrollment"]
merged["rate_per_1000"] = merged["expected_graduates"] / merged["total_enrollment"] * 1000
merged["school_group"] = merged["inst_name"].apply(
    lambda x: "UCLA" if x == "University of California-Los Angeles" else "Other Schools"
)

trend_data = merged.copy()


# In[15]:


# App layout
app = Dash(__name__)

server = app.server

app.layout = html.Div(
    children=[
        html.Div([
            html.H2("UCLA Graduate Earnings Dashboard"),

            html.Label("Earnings Horizon:"),
            dcc.RadioItems(
                id='earnings_horizon',
                options=[
                    {'label': '6-Year', 'value': '6_yr_median_earnings'},
                    {'label': '10-Year', 'value': '10_yr_median_earnings'}
                ],
                value='6_yr_median_earnings',
                inline=True
            ),

            dcc.Graph(id="scatter_plot", style={"width": "100%"}),

            html.Hr(),

            html.Label("Compare UCLA with another institution:"),
            dcc.Dropdown(
                id='comparison_school',
                options=school_options,
                value="University of Southern California"
            ),
            dcc.Graph(id="bar_plot", style={"width": "100%"}),

            html.Label("Select Field of Study:"),
            dcc.Dropdown(
                id='field_selector',
                options=[{'label': field, 'value': field} for field in sorted(trend_data['general_field'].unique())],
                value='Engineering'
            ),
            dcc.Graph(id='trend_plot', style={"width": "100%"})
        ])
    ],
    style={
        "maxWidth": "100%",
        "overflowX": "hidden",
        "overflowY": "auto",
        "padding": "20px",
        "boxSizing": "border-box"
    }
)

@app.callback(
    Output("scatter_plot", "figure"),
    Input("earnings_horizon", "value")
)
def update_scatter_plot(horizon):
    fig = px.scatter(
        enrollment,
        x="log_total_enrollment",
        y=horizon,
        color="school_group",
        size="total_enrollment",
        hover_name="school_name", 
        hover_data={
            "school_group": True,
            "total_enrollment": True,
            "log_total_enrollment": False,  
            horizon: True
        },
        title=f"Log Enrollment vs {horizon.replace('_', ' ').title()}",
        labels={
            horizon: "Median Earnings",
            "log_total_enrollment": "Log Enrollment",
            "total_enrollment": "Total Enrollment",
            "school_group": "Group"
        },
        color_discrete_map={
            "UCLA": "red", "Other UCs": "green", "CSU": "orange", "Other Schools": "blue"
        }
    )
    fig.update_traces(marker=dict(opacity=0.6))
    fig.add_traces(
        px.scatter(enrollment, x="log_total_enrollment", y=horizon, trendline="ols").data[1:]
    )
    return fig


@app.callback(
    Output("bar_plot", "figure"),
    Input("comparison_school", "value")
)
def update_bar_plot(school):
    target_unitids = [110404, 123961, 130794, 131469, 131496, 135726, 139658,
                      144050, 147767, 152080, 160755, 162928, 164924, 164988,
                      165015, 166027, 166683, 167358, 168148, 179867, 182670,
                      186131, 190150, 190415, 193900, 194824, 195030, 196413,
                      198419, 199847, 201645, 211440, 215062, 217156, 221999,
                      227757, 243744]
    
    UCLA_UNITID = 110662 
    
    # Combine UCLA with comparison school
    selected_unitids = target_unitids + [UCLA_UNITID]
    
    # Filter the data
    df = grad_numbers_data[
        grad_numbers_data["unitid"].isin(selected_unitids) &
        grad_numbers_data["gen_cip_code"].isin([11, 14, 42, 26, 45]) &
        grad_numbers_data["inst_name"].isin(["University of California-Los Angeles", school])
    ]



    if df.empty:
        return px.bar(title=f"No data available for {school} in selected fields.")

    df = df.groupby(["general_field", "inst_name"])["expected_graduates"].mean().reset_index()
    df.rename(columns={"inst_name": "Institution", "expected_graduates": "Avg Expected Graduates"}, inplace=True)

    fig = px.bar(
        df,
        x="general_field",
        y="Avg Expected Graduates",
        color="Institution",
        barmode="group",
        title=f"Expected Graduates by Field: UCLA vs {school}",
        labels={"general_field": "General Field"},
        height=600  
    )
    
    # Rotate x-axis labels and wrap long ones to prevent spillover
    fig.update_layout(
        xaxis_tickangle=-30,     # tilted upward instead of downward
        margin=dict(t=60, b=160), # more space for x labels
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    fig.update_xaxes(tickfont=dict(size=10), tickvals=df["general_field"].unique())

    return fig

@app.callback(
    Output("trend_plot", "figure"),
    Input("field_selector", "value")
)
def update_trend_plot(selected_field):
    field_df = trend_data[trend_data["general_field"] == selected_field]

    grouped = (
        field_df.groupby(["year", "school_group"])["expected_graduates"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        grouped,
        x="year",
        y="expected_graduates", 
        color="school_group",
        markers=True,
        title=f"Expected Graduates Over Time: {selected_field}",
        labels={
            "year": "Year",
            "expected_graduates": "Expected Graduates",
            "school_group": "School"
        },
        color_discrete_map={
            "UCLA": "blue",
            "Other Schools": "gold"
        }
    )
    
    fig.update_traces(line=dict(width=3))  # Thicken lines
    
    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis=dict(tickmode="linear", tick0=2004, dtick=2),
        yaxis=dict(title="Expected Graduates", gridcolor="#eeeeee"),
        legend_title="Institution"
    )
    return fig





if __name__ == "__main__":
    app.run_server(debug=True)

