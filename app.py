#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# In[2]:


# Load data
grad_numbers_data = pd.read_csv("Graduates_by_CIP_and_institution_2022.csv")
earnings_data = pd.read_csv("earnings_and_enrollment_by_institution.csv")

ucla_bach = pd.read_csv("ucla_bachelor_data.csv")


# Define institution groups
UCLA_UNITID = 110662
non_UCLA_UCs = [110635, 110680, 110644, 110653, 110705, 110714, 110671]
cal_states = [110486, 110495, 110510, 110529, 110538, 110547, 110556, 110565, 110574,
              110583, 110592, 110608, 110617, 111188, 115755]

# Tag schools
earnings_data["UCLA"] = earnings_data["unitid"] == UCLA_UNITID
earnings_data["non_UCLA_UC"] = earnings_data["unitid"].isin(non_UCLA_UCs)
earnings_data["CSU"] = earnings_data["unitid"].isin(cal_states)

# Filter 
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
comparison_unitids = [100663, 100751, 100858, 104151, 104179, 106397, 110565, 110583, 110635, 110644,
                     110653, 110662, 110671, 110680, 110705, 110714, 122409, 126614, 126775, 126818, 
                     129020, 130943, 133951, 134097, 134130, 139755, 139959, 145600, 145637, 145813, 
                     151351, 153603, 153658, 155317, 155399, 157085, 159391, 163268, 163286, 166629,
                     170976, 171100, 171128, 174066, 176017, 176080, 178396, 178411, 181464, 183044,
                     185828, 186371, 186380, 186399, 190567, 196060, 196079, 196088, 196097, 196103,
                     199120, 199193, 201885, 204024, 204796, 204857, 207388, 207500, 209542, 209551,
                     215293, 216339, 217484, 217882, 218663, 221759, 225511, 228723, 228778, 228787, 
                     230764, 231174, 231624, 232186, 232423, 233921, 234030, 234076, 236939, 236948,
                     240444, 243780]
valid_schools = grad_numbers_data[grad_numbers_data["unitid"].isin(comparison_unitids)]
school_options = [{'label': name, 'value': name} for name in sorted(valid_schools["inst_name"].unique()) if name != "University of California-Los Angeles"]




# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


top_public_unitids = [
    110635,  # UC Berkeley
    110644,  # UC Davis
    110653,  # UC Irvine
    110680,  # UC San Diego
    110705,  # UC Santa Barbara
    134130,  # University of Florida
    139959,  # University of Georgia
    145637,  # UIUC
    199120,  # UNC Chapel Hill
    234076   # UVA
]

UCLA_UNITID = 110662

top_public_df = earnings_data[
    earnings_data["unitid"].isin(top_public_unitids + [UCLA_UNITID])
].copy()

top_public_df["is_ucla"] = top_public_df["unitid"] == UCLA_UNITID
top_public_df = top_public_df.rename(columns={
    "school_name": "inst_name",
    "10_yr_median_earnings": "earn_10yr"
})


# In[7]:


top_national_unitids = [
    166683,  # MIT
    166027,  # Harvard
    243744,  # Stanford
    110404,  # Caltech
    144050,  # UChicago
    215062,  # UPenn
    186131,  # Princeton
    130794,  # Yale
    190415,  # Cornell
    190150,  # Columbia
    110635,  # UC Berkeley
    110662   # UCLA
]

top_national_df = earnings_data[
    earnings_data["unitid"].isin(top_national_unitids)
].copy()

top_national_df["is_ucla"] = top_national_df["unitid"] == UCLA_UNITID
top_national_df = top_national_df.rename(columns={
    "school_name": "inst_name",
    "10_yr_median_earnings": "earn_10yr"
})


# In[8]:


compare_df = pd.read_csv("compare_df.csv")
compare_df["is_ucla"] = compare_df["instnm"] == "University of California-Los Angeles"


# In[9]:


df_public = pd.read_csv("compare_all_public.csv")
df_elite = pd.read_csv("compare_all_elite.csv")
df_ca = pd.read_csv("compare_all_ca.csv")

df_public["group"] = "Top Public Schools"
df_elite["group"] = "Elite Universities"
df_ca["group"] = "California Schools"

# Combine into one DataFrame
all_data = pd.concat([df_public, df_elite, df_ca], ignore_index=True)


# In[10]:


def generate_dotplot(df, program, group_label):
    df_filtered = df[(df["program"] == program) & (df["group"] == group_label)]

    # Collapse duplicate institutions using median
    summary_df = df_filtered.groupby("institution", as_index=False)["EARN_MDN_5YR"].median()
    summary_df["is_ucla"] = summary_df["institution"].apply(lambda x: "UCLA" if x == "UCLA" else "Other")

    # Sort institution order by earnings DESC
    summary_df = summary_df.sort_values("EARN_MDN_5YR", ascending=False)
    institution_order = summary_df["institution"].tolist()

    fig = px.scatter(
        summary_df,
        x="institution",
        y="EARN_MDN_5YR",
        color="is_ucla",
        color_discrete_map={"UCLA": "steelblue", "Other": "gray"},
        title=f"{program}: UCLA vs {group_label}",
        labels={"EARN_MDN_5YR": "5-Year Median Earnings"},
    )

    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=30),
        height=450,
        xaxis=dict(categoryorder="array", categoryarray=institution_order)
    )
    return fig


# In[20]:


# App layout
app = Dash(__name__)

server = app.server

app.layout = html.Div(
    children=[
        html.Div([
            html.H2("UCLA Graduate Earnings Dashboard"),

            html.H3("Earnings vs. Enrollment"),
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

            
            html.H3("Enrollment vs. Other Schools"),
            html.Label("Select Another Institution:"),
            dcc.Dropdown(
                id='comparison_school',
                options=school_options,
                value="University of California-Berkeley"
            ),
            dcc.Graph(id="bar_plot", style={"width": "100%"}),

            html.H3("Enrollment Over Time"),
            html.Label("Select Field of Study:"),
            dcc.Dropdown(
                id='field_selector',
                options=[{'label': 'All Fields', 'value': 'ALL'}] +
                        [{'label': field, 'value': field} for field in sorted(trend_data['general_field'].unique())],
                value='ALL'
            ),
            dcc.Graph(id='trend_plot', style={"width": "100%"}),


            html.H3("Earnings vs. Debt by Program:"),
            dcc.RadioItems(
                id="debt_earnings_horizon",
                options=[
                    {"label": "1 Year", "value": "EARN_MDN_1YR"},
                    {"label": "5 Years", "value": "EARN_MDN_5YR"}
                ],
                value="EARN_MDN_1YR",
                inline=True
            ),

            dcc.Graph(id="debt_earnings_plot", style={"width": "100%"}),

            
            html.H3("Top vs Bottom Majors by Earnings"),
            dcc.Graph(id="earnings_combined_bar", style={"width": "100%"}),

            html.H3("Earnings Comparison: UCLA vs. Top Public Universities"),
            dcc.Graph(id="public_comparison"),
            
            html.H3("Earnings Comparison: UCLA vs. Top National Universities"),
            dcc.Graph(id="national_comparison"),

            html.H3("In-State Tuition Comparison"),
            dcc.Graph(id="tuition_comparison"),

            html.H3("Undergraduate Enrollment Comparison"),
            dcc.Graph(id="enrollment_comparison"),

            html.H3("Earnings by Field of Study: UCLA vs Other Institutions"),
            
            html.Label("Select a Program:"),
            dcc.Dropdown(
                id="program_selector",
                options=[{"label": prog, "value": prog} for prog in sorted(all_data["program"].unique())],
                value="Computer Science",
                style={"width": "60%"}
            ),
            
            html.Div(id="plot_container")

                    
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
    target_unitids = [100663, 100751, 100858, 104151, 104179, 106397, 110565, 110583, 110635, 110644,
                     110653, 110662, 110671, 110680, 110705, 110714, 122409, 126614, 126775, 126818, 
                     129020, 130943, 133951, 134097, 134130, 139755, 139959, 145600, 145637, 145813, 
                     151351, 153603, 153658, 155317, 155399, 157085, 159391, 163268, 163286, 166629,
                     170976, 171100, 171128, 174066, 176017, 176080, 178396, 178411, 181464, 183044,
                     185828, 186371, 186380, 186399, 190567, 196060, 196079, 196088, 196097, 196103,
                     199120, 199193, 201885, 204024, 204796, 204857, 207388, 207500, 209542, 209551,
                     215293, 216339, 217484, 217882, 218663, 221759, 225511, 228723, 228778, 228787, 
                     230764, 231174, 231624, 232186, 232423, 233921, 234030, 234076, 236939, 236948,
                     240444, 243780]
 
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
        height=600,
        color_discrete_map={
            "University of California-Los Angeles": 'blue',
            school: 'red'
        }
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
    if selected_field == "ALL":
        field_df = trend_data.copy()
        title = "Expected Graduates Over Time: All Fields"
    else:
        field_df = trend_data[trend_data["general_field"] == selected_field]
        title = f"Expected Graduates Over Time: {selected_field}"

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
        title=title,
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

    fig.update_traces(line=dict(width=3))

    fig.update_layout(
        height=500,
        template="plotly",
        xaxis=dict(tickmode="linear", tick0=2004, dtick=2),
        yaxis=dict(title="Expected Graduates", gridcolor="#eeeeee"),
        legend_title="Institution"
    )

    return fig


@app.callback(
    Output("debt_earnings_plot", "figure"),
    Input("debt_earnings_horizon", "value")
)
def update_debt_earnings_plot(horizon):
    df = ucla_bach.copy()
    df = df.dropna(subset=["DEBT_ALL_PP_ANY_MDN", horizon, "CIPDESC"])

    # Create base scatter plot with trendline
    fig = px.scatter(
        df,
        x="DEBT_ALL_PP_ANY_MDN",
        y=horizon,
        hover_name="CIPDESC",
        trendline="ols",
        #title="Debt vs. Early Earnings by Program",
        labels={
            "DEBT_ALL_PP_ANY_MDN": "Median Debt at Graduation",
            horizon: "Median Earnings",
        }
    )

    # Style markers (first trace)
    fig.data[0].marker = dict(
        color="blue",  
        size=12,          
        line=dict(width=1, color='white'),
        opacity=0.75
    )
    fig.data[0].hovertemplate = "<b>%{hovertext}</b><br>Debt: %{x:$,.0f}<br>Earnings: %{y:$,.0f}<extra></extra>"

    # Style trendline (second trace)
    fig.data[1].line.color = "red"
    fig.data[1].line.width = 3
    fig.data[1].line.dash = "solid"
    fig.data[1].name = "Linear Trend"
    fig.data[1].showlegend = False

    # Layout styling
    fig.update_layout(
        height=600,
        template="plotly",
        font=dict(size=14),
        xaxis=dict(
            tickformat="$,.0f",
            title=dict(text="Median Debt at Graduation", font=dict(size=16))
        ),
        yaxis=dict(
            tickformat="$,.0f",
            title=dict(text="Median Earnings", font=dict(size=16)),
            gridcolor="#eeeeee"
        ),
        title=dict(font=dict(size=20, family="Arial", color="#2C3E50")),
        margin=dict(l=60, r=30, t=60, b=60)
    )

    return fig



@app.callback(
    Output("earnings_combined_bar", "figure"),
    Input("earnings_horizon", "value")  # dummy input just to render once
)
def render_combined_bar(_):
    df = ucla_bach.copy()
    df = df.dropna(subset=["EARN_MDN_1YR", "EARN_MDN_5YR"])

    # Get top and bottom 10 majors by 1-year earnings
    top10 = df.sort_values("EARN_MDN_1YR", ascending=False).head(10).copy()
    bottom10 = df.sort_values("EARN_MDN_1YR", ascending=True).head(10).copy()

    # Combine and tag
    combined = pd.concat([top10, bottom10], ignore_index=True)
    combined["RankGroup"] = ["Top 10"] * 10 + ["Bottom 10"] * 10

    # Desired final y-axis order:
    # Top 10 descending, Bottom 10 descending
    top10_order = top10.sort_values("EARN_MDN_1YR", ascending=False)["CIPDESC"].tolist()
    bottom10_order = bottom10.sort_values("EARN_MDN_1YR", ascending=False)["CIPDESC"].tolist()
    ordered_majors = top10_order + bottom10_order

    # Set ordered categorical
    combined["CIPDESC"] = pd.Categorical(
        combined["CIPDESC"],
        categories=ordered_majors,
        ordered=True
    )

    # Convert to long format
    long_df = pd.melt(
        combined,
        id_vars=["CIPDESC"],
        value_vars=["EARN_MDN_1YR", "EARN_MDN_5YR"],
        var_name="Time",
        value_name="Earnings"
    )
    long_df["Time"] = long_df["Time"].map({
        "EARN_MDN_1YR": "1 Year",
        "EARN_MDN_5YR": "5 Year"
    })

    long_df = long_df.sort_values("CIPDESC", ascending=False)

    # Plot
    fig = px.bar(
        long_df,
        x="Earnings",
        y="CIPDESC",
        color="Time",
        barmode="group",
        orientation="h",
        color_discrete_map={"1 Year": "skyblue", "5 Year": "salmon"},
        #title="Median Earnings: 1-Year vs 5-Year After Graduation <br> for Highest and Lowest Earning 10 Majors After 1-Year",
        labels={"CIPDESC": "Major"}
    )

    # Layout styling
    fig.update_layout(
        height=800,
        template="plotly",
        font=dict(size=13),
        margin=dict(l=120, r=30, t=70, b=50),
        xaxis_tickformat="$,.0f"
    )

    return fig


@app.callback(
    Output("public_comparison", "figure"),
    Input("public_comparison", "id")
)
def render_public_bar(_):
    df = earnings_data[
        earnings_data["unitid"].isin(top_public_unitids + [UCLA_UNITID])
    ].copy()

    # Rename for consistency
    df = df.rename(columns={
        "school_name": "inst_name",
        "10_yr_median_earnings": "earn_10yr"
    })

    # Drop rows with missing earnings
    df = df.dropna(subset=["earn_10yr"])

    # Add UCLA tag again
    df["is_ucla"] = df["unitid"] == UCLA_UNITID

    # Sort by 10-year earnings descending
    df = df.sort_values("earn_10yr", ascending=False)

    # Enforce category order for y-axis
    df["inst_name"] = pd.Categorical(df["inst_name"], categories=df["inst_name"], ordered=True)

    fig = px.bar(
        df,
        x="earn_10yr",
        y="inst_name",
        orientation="h",
        color="is_ucla",
        color_discrete_map={True: "steelblue", False: "lightgray"},
        labels={"earn_10yr": "10-Year Median Earnings", "inst_name": "Institution"},
        #title="UCLA vs. Top Public Universities"
    )

    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=df["inst_name"].tolist()),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=120, r=30, t=60, b=40),
        xaxis_tickformat="$,.0f"
    )

    return fig






@app.callback(
    Output("national_comparison", "figure"),
    Input("national_comparison", "id")  # dummy trigger
)
def render_national_bar(_):
    df = top_national_df.copy()

    # Sort by 10-year earnings descending
    df = df.sort_values("earn_10yr", ascending=False)

    # Enforce category order on y-axis
    df["inst_name"] = pd.Categorical(df["inst_name"], categories=df["inst_name"], ordered=True)

    fig = px.bar(
        df,
        x="earn_10yr",
        y="inst_name",
        orientation="h",
        color="is_ucla",
        color_discrete_map={True: "steelblue", False: "lightgray"},
        labels={"earn_10yr": "10-Year Median Earnings", "inst_name": "Institution"},
        #title="UCLA vs. Top National Universities"
    )

    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=df["inst_name"].tolist()),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=120, r=30, t=60, b=40),
        xaxis_tickformat="$,.0f"
    )

    return fig


@app.callback(
    Output("tuition_comparison", "figure"),
    Input("tuition_comparison", "id")  # dummy input
)
def render_tuition_plot(_):
    df = compare_df.sort_values("tuitionfee_in", ascending=False)
    fig = px.bar(
        df,
        x="tuitionfee_in",
        y="instnm",
        orientation="h",
        color="is_ucla",
        color_discrete_map={True: "darkgreen", False: "lightgray"},
        labels={"tuitionfee_in": "In-State Tuition ($)", "instnm": "Institution"},
        #title="In-State Tuition Comparison"
    )
    fig.update_layout(showlegend=False, template="plotly_white", margin=dict(l=120, r=30, t=60, b=40))
    return fig

@app.callback(
    Output("enrollment_comparison", "figure"),
    Input("enrollment_comparison", "id")
)
def render_enrollment_plot(_):
    df = compare_df.copy()
    df = df.sort_values("ugds", ascending=False)

    fig = px.bar(
        df,
        x="ugds",
        y="instnm",
        orientation="h",
        color="is_ucla",
        color_discrete_map={True: "purple", False: "lightgray"},
        labels={"ugds": "Undergraduate Enrollment", "instnm": "Institution"},
        #title="Undergraduate Enrollment Comparison"
    )

    # Explicit y-axis order by sorted values
    fig.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=df["instnm"].tolist()),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=120, r=30, t=60, b=40)
    )

    return fig


@app.callback(
    Output("plot_container", "children"),
    Input("program_selector", "value")
)
def update_plots(selected_program):
    return [
        dcc.Graph(figure=generate_dotplot(all_data, selected_program, "Top Public Schools")),
        dcc.Graph(figure=generate_dotplot(all_data, selected_program, "Elite Universities")),
        dcc.Graph(figure=generate_dotplot(all_data, selected_program, "California Schools")),
    ]







if __name__ == "__main__":
    app.run_server(debug=True)


# In[ ]:




