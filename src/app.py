#!/usr/bin/env python
# coding: utf-8

import dash
from dash.dependencies import Input, Output, State
from dash import dash_table

from dash.dash_table.Format import Group, Format
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html

import plotly.express as px
import pandas as pd

import plotly.graph_objects as go
from plotly_calplot import calplot
import plotly.figure_factory as ff

import numpy

from math import factorial

# external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cerulean/bootstrap.min.css"]

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.COSMO])
server = app.server

df = pd.read_pickle('data_la_mundial_2022.pickle')



def facturacion_detalle(data):
	map_day_name = {
        'Monday'   :'01_Monday',
        'Tuesday'  :'02_Tuesday',
        'Wednesday':'03_Wednesday',
        'Thursday' :'04_Thursday',
        'Friday'   :'05_Friday',
        'Saturday' :'06_Saturday',
        'Sunday'   :'07_Sunday'
	}
	map_month_name = {
        'January':'01_January',
        'February':'02_February',
        'March':'03_March',
        'April':'04_April',
        'May':'05_May',
        'June':'06_June',
        'July':'07_July',
        'August':'08_August',
        'September':'09_September',
        'October':'10_October',
        'November':'11_November',
        'December':'12_December'
	}
	df = data            .reset_index()            .groupby('fecha')            .agg({'precio_total_con_iva':sum})            .assign(date = lambda df_: df_.index.date)            .groupby('date')            .agg({'precio_total_con_iva':sum})            .asfreq('D')            .assign(day_name = lambda df_: df_.index.day_name())            .assign(month_name = lambda df_: df_.index.month_name())            .assign(day = lambda df_: df_.index.day)
	df.day_name   = df.day_name.map(map_day_name)
	df.month_name = df.month_name.map(map_month_name)

    # ------------------------------------------------
	condicion_tercio_mes = [
        (df.day <= 10),
        (df.day >10) & (df.day<=20),
        (df.day >20)
	]

	resultado_t = ['tercio_1', 'tercio_2', 'tercio_3']
    # ------------------------------------------------
	condicion_semana_mes = [
        (df.day <= 7),
        (df.day >7)  & (df.day<=14),
        (df.day >14) & (df.day<=21),
        (df.day >21) & (df.day<=28),
        (df.day >28)
	]

	resultado_s = ['sem_1', 'sem_2', 'sem_3', 'sem_4', 'sem_5']
    # ------------------------------------------------

	df['tercio_mes'] = np.select(condicion_tercio_mes, resultado_t, default=-999)
	df['semana_mes'] = np.select(condicion_semana_mes, resultado_s, default=-999)

	return df
# =======================================================================================
def consumo_detalle(data):	
	df = data.reset_index().groupby(['fecha','familia']).agg({'cantidad':sum}).unstack(level=1).fillna(0)
    	df.columns = df.columns.droplevel(0)
    	return df
# =======================================================================================
def pv_facturacion(data, rows, cols):
    
    	df = pd.pivot_table(data   = facturacion,
                        index  = rows,
                        columns= cols,
                        values ='precio_total_con_iva',
                        aggfunc=lambda x: np.round(np.sum(x)/1000,2)).fillna(0)
    	return df
# =======================================================================================
def grafico_hm(df):
    	# 'RdBu_r', 'inferno', 'Bluered_r', 'px.colors.cyclical.IceFire'
    	scale_color = px.colors.cyclical.IceFire
    	fig = px.imshow(df, text_auto=True, aspect="auto",color_continuous_scale=scale_color, origin='upper')
    	return fig
# =======================================================================================
def corr_hm(df):
    	# 'RdBu_r', 'inferno'
    	fig = px.imshow(df, text_auto=True, aspect="auto",color_continuous_scale='RdBu_r', origin='upper')
    	return fig


facturacion = facturacion_detalle(df)
consumo = consumo_detalle(df)
    
pv_fact_mes_dia = pv_facturacion(facturacion, 'day_name', 'month_name')
pv_fact_dia_mes = pv_facturacion(data, 'day', 'month_name')
pv_fact_sem_mes = pv_facturacion(data, 'semana_mes', 'month_name')

# =======================================================================================

def render_table(df):    

    	columns=[
                {
                    "name"      : i,
                    "id"        : i,
                    "deletable" : True,
                    "selectable": True,
                    "hideable"  : True
                } for i in df.columns]
    
    	font_size = 12
    
    	table_content =  dash_table.DataTable(
                                id                ="data_table",
                                columns           =columns,
                                data              =df.to_dict("records"),
                                export_format     ='xlsx',
                                style_header      ={
                                    "textDecoration"     : "underline",
                                    "textDecorationStyle": "dotted",
                                    'backgroundColor'    : '#193a6e',
                                    'padding'            : '12px',
                                    'color'              : '#FFFFFF',
                                    'width'              : 'auto'},
                                tooltip_delay     =0,
                                tooltip_duration  =None,
                                page_size = 30,
                                filter_action     ="native",
                                row_deletable     =True,
                                column_selectable ="multi",
#                                 hidden_columns    =None,
                                style_table       ={
                                    'height': 800},
                                fixed_rows        ={
                                    "headers": True, 
                                    "data"   : 0},
                                style_cell        ={
                                    'minWidth': 95, 
                                    'maxWidth': 100, 
                                    'width'   : 100
                                },
                                style_cell_conditional=[
                                    {
                                        'if'       : {'column_id': c},
                                        'textAlign': 'right',
                                        'font-size': f'{font_size}px'
                                    } for c in df.columns
                                ],
                                style_data       ={
                                    'whiteSpace': 'normal',
                                    'height'    : 'auto',
                                    'font-size' : f'{font_size}px'
                                }
                            )
    
    	return table_content
# =======================================================================================
card_table = dbc.Row([
    dbc.Card([
        render_table(facturacion)
    ])
])



card_params = {'card_color':"dark",
               'outline'  : True}
# =======================================================================================
row_1 = dbc.Row([
    dbc.Card([
        dcc.Graph(id='fig-heatmap-dia-mes', figure = grafico_hm(pv_fact_dia_mes))
    ], color = card_params['card_color'], outline = card_params['outline']
    )
])

row_2 = dbc.Row([
    dbc.Card([
        dcc.Graph(id='fig-heatmap-corr-cons', figure = corr_hm(consumo.corr()))
    ], color = card_params['card_color'], outline = card_params['outline']
    )
])



app.layout = html.Div(
    [
        dbc.Container(
            [
                card_table,
                row_1,
                row_2              
            ],
            fluid=True
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False)

