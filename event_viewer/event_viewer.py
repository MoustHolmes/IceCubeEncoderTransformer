import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc

from scipy.spatial import ConvexHull


def load_attention_weights(event_number, db_path ):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Query the row corresponding to the event_number
    query = f"SELECT * FROM attention_weights WHERE event_number = {event_number}"
    df = pd.read_sql(query, conn, index_col='event_number')
    
    # Close the SQLite database connection
    conn.close()
    
    # Convert the matrix columns from BLOB to NumPy arrays and reconstruct the dictionary
    matrices = {}
    for column in df.columns:
        attn_type, layer_no, head_no = column.rsplit('_', 2)
        layer_no = int(layer_no)
        head_no = int(head_no)
        matrix_data = df.at[event_number, column]
        if matrix_data is not None:
            size = int(np.sqrt(len(matrix_data) // 8))
            matrix = np.frombuffer(matrix_data, dtype=np.float64).reshape((size, size))
            matrices[(attn_type, layer_no, head_no)] = matrix
    
    return matrices

def add_attention_scores(df, attention_dict):
    new_columns = {}
    
    # 1. Compute the attention scores for each matrix
    for (attn_type, layer, head), matrix in attention_dict.items():
        if attn_type == "rel_attn":
            attention_scores = matrix.mean(axis=0)
        elif attn_type == "attn":
            attention_scores = matrix[1:,1:].mean(axis=0)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        
        # Check if any attention scores themselves are NaN
        if np.isnan(attention_scores).any():
            print(f"Warning: NaN values found in attention scores for {attn_type}_{layer}_{head}")

        # Store the attention scores in the new_columns dictionary
        column_name = f"{attn_type}_{layer}_{head}_mean"
        new_columns[column_name] = attention_scores

    # 2. Compute the mean attention scores across heads for each attn_type and layer
    for attn_type in set([key[0] for key in attention_dict.keys()]):
        for layer in set([key[1] for key in attention_dict.keys() if key[0] == attn_type]):
            columns_of_interest = [col for col in new_columns if col.startswith(f"{attn_type}_{layer}_") and col.endswith("_mean")]
            
            # Check if columns_of_interest is empty
            if not columns_of_interest:
                print(f"Warning: No columns found for {attn_type}_{layer}. This could be due to a mismatch in naming or missing data.")
                continue
            new_columns[f"{attn_type}_{layer}_mean"] = np.nanmean([values for col, values in new_columns.items() if col in columns_of_interest], axis=0)

    # 3. Compute the mean attention scores across layers and heads for each attn_type
    for attn_type in set([key[0] for key in attention_dict.keys()]):
        columns_of_interest = [col for col in df.columns if col.startswith(f"{attn_type}_")]
        new_columns[f"{attn_type}_mean"] = df[columns_of_interest].mean(axis=1)

    # 4. For attn_type "attn", extract cls token attention and compute means
    if "attn" in [key[0] for key in attention_dict.keys()]:
        for (attn_type, layer, head), matrix in attention_dict.items():
            if attn_type == "attn":
                cls_attention_scores = matrix[1:, 0]
                column_name = f"{attn_type}_{layer}_{head}_cls"
                new_columns[column_name] = cls_attention_scores

        for layer in set([key[1] for key in attention_dict.keys() if key[0] == "attn"]):
            columns_of_interest = [col for col in new_columns if col.startswith(f"attn_{layer}_") and col.endswith("_cls")]
            new_columns[f"attn_{layer}_cls_mean"] = np.nanmean([values for col, values in new_columns.items() if col in columns_of_interest], axis=0)

        columns_of_interest = [col for col in new_columns if col.startswith(f"attn_") and col.endswith("_cls")]
        new_columns[f"attn_cls_mean"] = np.nanmean([values for col, values in new_columns.items() if col in columns_of_interest], axis=0)

    # Convert the new_columns dictionary to a DataFrame and concatenate to the original DataFrame
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return new_df


def load_Truth_from_sql(event_no,db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                *
            FROM 
                Truth
            WHERE
                event_no = """+ str(event_no)+';', conn)
    return data

def load_all_Truth_from_sql(db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                *
            FROM 
                Truth
            ;""", conn)
    return data

def load_SplitInIcePulses_from_sql(event_no,db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                *
            FROM 
                SplitInIcePulses
            WHERE
                event_no = """+ str(event_no)+';', conn)
    return data

def load_SplitInIcePulses_dynedge_v2_Pulses_from_sql(event_no,db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                *
            FROM 
                SplitInIcePulses_dynedge_v2_Pulses
            WHERE
                event_no = """+ str(event_no)+';', conn)
    return data


def load_noise_from_sql(event_no,db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                dom_time, truth_flag
            FROM 
                SplitInIcePulses_TruthFlags
            WHERE
                event_no = """+ str(event_no)+';', conn)
    return data

def load_noise_pred_from_sql(event_no,db_path):
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query("""
            SELECT
                dom_time, truth_flag
            FROM 
                SplitInIcePulses_dynedge_v2_Predictions
            WHERE
                event_no = """+ str(event_no)+';', conn)
    return data

def create_beta_loss_df(path):
    df= pd.read_csv(path)
    alpha = df["alpha"]
    beta = df["beta"]
    df["mean"] = alpha / (alpha + beta)
    df["var"] = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    return df

def load_truth(db_path,inelasticity_pred_path):
    truth = load_all_Truth_from_sql(db_path)
    inelasticity_pred = create_beta_loss_df(inelasticity_pred_path)
    truth = truth.merge(inelasticity_pred, on="event_no")
    return truth

def load_data_noise(event_no,db_path):
    df = load_SplitInIcePulses_from_sql(event_no,db_path)
    noise_truth = load_noise_from_sql(event_no,db_path)
    noise_pred = load_noise_pred_from_sql(event_no,db_path)
    df["noise_flag"] = noise_truth["truth_flag"]
    df["noise_pred"] = noise_pred["truth_flag"]
    return df

def load_data_clean(event_no,db_path,attn_weights_db_path):
    df = load_SplitInIcePulses_dynedge_v2_Pulses_from_sql(event_no,db_path)
    attn_weights = load_attention_weights(event_no, attn_weights_db_path)
    df = add_attention_scores(df, attn_weights)
    return df

db_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/Upgrade_Data/sqlite3/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash.db"
event_no_list = [5794046, 6135033, 1107604, 637980, 6326499,3373387]

app = dash.Dash(__name__, external_stylesheets=[
    'https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/darkly/bootstrap.min.css'
    ],
    meta_tags=[{'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0'}]
    )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("IceCube event viewer",
                        className='text-center text-primary mb-4'),
                width=12)
    ),

    dbc.Row([

        dbc.Col([
            dcc.Input(
                id='database',
                type='text',
                value=db_path,
                placeholder=db_path
            ),
            dcc.Dropdown(id='event_no', multi=False, value=3373387,
                         options=[{'label':x, 'value':x}
                                  for x in sorted(event_no_list)],
                         ),
            dcc.Graph(id='graph')
        ], width={'size':10, 'offset':0, 'order':1},
        #    xs=12, sm=12, md=12, lg=5, xl=5
        )
    ], justify='start'),  # Horizontal:start,center,end,between,around
    dcc.Store(id='camera-state', storage_type='session'),
    
], fluid=True)



@app.callback(
    Output("graph", "figure"), 
    Input("database", "value"),
    Input("event_no", "value"),
    Input('camera-state', 'data'),
    prevent_initial_call=True
)
def update_bar_chart(database, event_no, camera_data):
    
    # '/groups/icecube/petersen/GraphNetDatabaseRepository/nmo_analysis/data/140028_upgrade_NuMu/merged_140021.db'
    attn_weights_db_path = "/groups/icecube/moust/work/IceCubeEncoderTransformer/logs/train/runs/2023-09-24_18-07-18/attention_weights.db"
    inelasticity_pred_path = "/groups/icecube/moust/work/IceCubeEncoderTransformer/logs/train/runs/2023-09-24_18-07-18/beta_predictions.csv"
    dom_info_path = "/groups/icecube/moust/work/IceCubeEncoderTransformer/notebooks/dom_info.csv"
    # event_no =3373387
    # gloabal_truth = load_truth(db_path,inelasticity_pred_path)
    truth = load_Truth_from_sql(event_no,db_path)
    # data_noise = load_data_noise(event_no,db_path)
    data_clean = load_data_clean(event_no,db_path,attn_weights_db_path)
    dom_info = pd.read_csv(dom_info_path)

    # clean_mask = noise_pred["truth_flag"]>0.7
    color_options = ["attn_cls_mean","dom_time", "dom_x"]
    data_clean.sort_values(by=['dom_time'], inplace=True)
    x, y, z = data_clean['dom_x'], data_clean['dom_y'], data_clean['dom_z']


    # Define the color scale
    colorscale = "rdylgn"

    # direction vector length
    r=400

    # Create two separate traces for DOM hits with different symbols
    trace_cleaned = go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        name="DOM hit",
        # hoverinfo=noise_pred.loc[mask_truth, 'truth_flag'],
        marker=dict(
            size=data_clean[ 'charge'] * 10,
            color=data_clean["attn_cls_mean"],  # specify the color here
            colorscale=colorscale,
            symbol='circle',
            line=dict(width=0),
            showscale=True,
        ),
        text = [f"charge:{charge:.2g}<br>dom_time:{dom_time:.2g}<br>DOM_type:{dom_type}<br>dom_number:{dom_number}<br>string:{string}"
            for dom_type, charge,dom_time,dom_number,string in data_clean[['dom_type','charge','dom_time','dom_number','string']].itertuples(index=False)]
    )

    # dom positions
    dom_position = go.Scatter3d(
        x=dom_info['dom_x'], 
        y=dom_info['dom_y'], 
        z=dom_info['dom_z'],
        hoverinfo= 'none',
        # visible='legendonly',
        mode='markers',
        name= 'Dom positions',
        marker=dict(
            size=1,
            color=dom_info['dom_type'],
            line=dict(width=0),
            opacity=0.4,
            colorscale='Reds'
            )
    )

    # interaction vertex and direction
    interaction_vertex = go.Scatter3d(
        x=truth['position_x'],
        y=truth['position_y'],
        z=truth['position_z'],
        mode='markers',
        name= 'Interaction vertex',
        marker=dict(
            size=10,
            color='orange',
            line=dict(width=0),
            opacity=0.8,
            symbol='cross',
            )
    )

    # direction vector from interaction vertex
    direction_vector = go.Scatter3d(
        x=[truth['position_x'].iloc[0], truth['position_x'].iloc[0] + r*np.cos(truth['azimuth'].iloc[0])*np.sin(truth['zenith'].iloc[0])],
        y=[truth['position_y'].iloc[0], truth['position_y'].iloc[0] + r*np.sin(truth['azimuth'].iloc[0])*np.sin(truth['zenith'].iloc[0])],
        z=[truth['position_z'].iloc[0], truth['position_z'].iloc[0] + r*np.cos(truth['zenith'].iloc[0])],
        mode='lines',  # Change mode to 'lines+markers'
        name='Direction vector',
        line=dict(color='orange', width=1)  # Set color and width of the line
    )


    # x y string positions
    unique_string_df = dom_info.groupby('string').first().reset_index()
    points = unique_string_df[["dom_x", "dom_y"]].to_numpy()
    hull = ConvexHull(points)

    # Extract the vertices of the Convex Hull
    vertices = points[hull.vertices]

    z_position = -600

    # Define the triangles for the Mesh3d trace
    num_vertices = len(vertices)
    i = np.zeros(num_vertices - 2, dtype=int)
    j = np.arange(1, num_vertices - 1, dtype=int)
    k = np.arange(2, num_vertices, dtype=int)

    # Add the filled shape in 3D space (on the XY plane) using Mesh3d
    detector_area_mesh = go.Mesh3d(
                            x=vertices[:, 0], 
                            y=vertices[:, 1], 
                            z=np.full(vertices.shape[0], z_position),
                            i=i, 
                            j=j, 
                            k=k,
                            name= 'Detector footprint',
                            color='darkblue',
                            opacity=0.2,     
                            # visible='legendonly',
                            hoverinfo='none'
                            )

    # Add all points, including those inside the Convex Hull, in 3D space (on the XY plane)
    string_pos = go.Scatter3d(x=unique_string_df["dom_x"], 
                            y=unique_string_df["dom_y"],
                            z=np.full(unique_string_df.shape[0], z_position),
                            mode='markers', 
                            name='Points',
                            #    hoverinfo='none',
                            marker=dict(
                                size=1, 
                                opacity=0.5, 
                                symbol='x',
                                ))

    fig = go.Figure(data=[detector_area_mesh, string_pos, trace_cleaned, dom_position, interaction_vertex, direction_vector ])


    # Create a list of unique dom_times
    unique_times = data_clean['dom_time'].unique()

    frames = [go.Frame(
        data=[
            go.Scatter3d(
                x=data_clean[data_clean['dom_time'] <= time]['dom_x'], 
                y=data_clean[data_clean['dom_time'] <= time]['dom_y'], 
                z=data_clean[data_clean['dom_time'] <= time]['dom_z']
            ),
        ],
        traces=[2],
        name=f'frame{idx}'
    ) for idx, time in enumerate(unique_times)]


    fig.update(frames=frames)


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
                }

    color_ranges = {opt: [data_clean[opt].min(), data_clean[opt].max()] for opt in color_options}


    dropdown = {
        "buttons": [
            {
                "args": [
                    {
                        "marker.color": [data_clean[opt].tolist()],
                        "marker.cmin": [color_ranges[opt][0]],
                        "marker.cmax": [color_ranges[opt][1]]
                    },
                    [2]
                ],
                "label": opt,
                "method": "restyle",
            }
            for opt in color_options
            ],
            "direction": "down",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
        }


    sliders = [
        {"pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [[f.name], frame_args(0)],
            "label": f"{time:.2g} ",  # Change the label to display the dom_time value
            "method": "animate",
            } for time, f in zip(unique_times, fig.frames)
        ]
    }
    ]

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 50, "easing": "linear"}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": True,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
            },
            dropdown,
        ],
        sliders=sliders,

    )


    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X (m)',
                range=[-571, 577],
                autorange=False,
            ),
            yaxis=dict(
                title='Y (m)',
                range=[-522, 510],
                autorange=False,
            ),
            zaxis=dict(
                title='Z (m)',
                range=[-750, 580],
                autorange=False,
            )
        ),
        margin=dict(
            l=0,
            r=100, 
            b=0, 
            t=50),
        width=1200,
        height=900,
        template = "plotly_dark", #plotly_white, plotly_dark
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),


        sliders = sliders
    )


    # Apply the stored camera state if it exists
    if camera_data and 'camera' in camera_data:
        fig.update_layout(scene_camera=camera_data['camera'])

    return fig



if __name__=='__main__':
    app.run_server(host = '0.0.0.0',debug=True,port=3000)