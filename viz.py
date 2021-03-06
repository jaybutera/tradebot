import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

#class Visualizer:
#    def __init__ (self):

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    #dcc.Input(id='my-id', value='initial value', type="text"),

    dcc.Graph(
        id='viz',
        figure={
            'data': [go.Scatter(
                    x= [i for i in range(len(data))],
                    y= [i for i in data],
                    mode='markers',
                    opacity = .9,
                    marker={
                        'size' : 15,
                        'line' : { 'width' : .5, 'color' : 'black'}
                        },
                    ) for data in []]
        }
    )
])

'''
@app.callback(
    Output(component_id='viz', component_property='data'),
    [Input(component_id='my-id', component_property='value')]
)
'''

def update_graph (prices, assets_db, usd_db, crypt_db):
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Graph(
            id='viz',
            figure={
                'data': [go.Scatter(
                        x= [i for i in range(len(data))],
                        y= [i for i in data],
                        mode='markers',
                        opacity = .9,
                        marker={
                            'size' : 15,
                            'line' : { 'width' : .5, 'color' : 'black'}
                            },
                        ) for data in [prices,assets_db,usd_db,crypt_db]]
            }
        )
    ])

app.run_server(debug=True)
