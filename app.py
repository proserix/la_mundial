import dash
from dash import Dash
# import jupyter_dash
from dash.dependencies import Input, Output, State
from dash import dash_table, no_update
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
from plotly.subplots import make_subplots

from dash.dash_table.Format import Group, Format
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html

import plotly.express as px
import pandas as pd

import plotly.graph_objects as go
from plotly_calplot import calplot
import plotly.figure_factory as ff

from plotly_calplot import calplot

import numpy as np
from datetime import date, datetime, timedelta
from math import factorial
# from tqdm import tqdm

import io
import base64


# ======================================================================
# font_size = 10
# export_format     ='xlsx'

# archivos = [
#     # 'la_mundial_ene_2022.xlsx',
#     # 'la_mundial_feb_may_2022.xlsx',
#     # 'la_mundial_jun_sep_2022.xlsx',
#     'la_mundial_oct_dic_2022.xlsx'
# ]
# # # feriados = pd.read_pickle('feriados_completo.pickle')
# # ======================================================================
def armarData(df):

    columns_string = [
        'fecha',
        'fecha_caja',
        'grupo_denom',
        'zona_denom',
        'numero',
        'tipo_doc',
        'denominacion'
    ]

    columns_float = [
        'cantidad',
        'precio_costo',
        'precio_lista',
        'precio_total_sin_iva',
        'precio_total_con_iva'
    ]

    tipo_datos = {
        c: 'string'if c in columns_string else 'float64' for c in columns_string + columns_float
    }

    columns_final = ['date', 'date_2'] + columns_float + \
        ['rentabilidad'] + columns_string[3:]

    df.columns = [col.lower() for col in df.columns]
    df = df[columns_string + columns_float]
    df = df.astype(tipo_datos)

    df['date_2'] = df.apply(lambda x: datetime.strptime(
        (x.fecha + ' ' + x.grupo_denom[-5:]), '%Y/%m/%d %H:%M'), axis=1)
    df['date'] = df.apply(lambda x: datetime.strptime(
        (x.fecha_caja + ' ' + x.grupo_denom[-5:]), '%Y/%m/%d %H:%M'), axis=1)
    df['rentabilidad'] = df.apply(
        lambda x: x.precio_total_sin_iva-x.precio_costo, axis=1)

    df[columns_string] = df[columns_string].fillna(
        'error').applymap(lambda x: x.lower().replace(' ', '_'))

    return df[columns_final]


familia_denominacion = pd.read_pickle('familia_denominacion.pickle')
familia_denominacion.columns = [col.lower()
                                for col in familia_denominacion.columns]
familia_denominacion = familia_denominacion.applymap(
    lambda x: x.lower().replace(' ', '_'))

# print(familia_denominacion.info())
# print(familia_denominacion.columns)
# print(familia_denominacion)


def insert_calendar_vals(df):

    data = (
        df.set_index('date')
        .assign(fecha=lambda df_: df_.index.date)
        .assign(year=lambda df_: df_.index.year)
        .assign(mes=lambda df_: df_.index.month)
        .assign(day=lambda df_: df_.index.day)
        .assign(day_name=lambda df_: df_.index.day_name())
        .assign(month_name=lambda df_: df_.index.month_name())
        .assign(hora=lambda df_: df_.index.hour)
        #         .assign(minutos = lambda df_: df_.index.minute)
    )
    # ==================================================================================================
    '''
        recodificar month_name y day_name
    '''
    map_day_name = {
        'Monday': '01_Monday',
        'Tuesday': '02_Tuesday',
        'Wednesday': '03_Wednesday',
        'Thursday': '04_Thursday',
        'Friday': '05_Friday',
        'Saturday': '06_Saturday',
        'Sunday': '07_Sunday'
    }

    map_month_name = {
        'January': '01_January',
        'February': '02_February',
        'March': '03_March',
        'April': '04_April',
        'May': '05_May',
        'June': '06_June',
        'July': '07_July',
        'August': '08_August',
        'September': '09_September',
        'October': '10_October',
        'November': '11_November',
        'December': '12_December'
    }
    data.day_name = data.day_name.map(map_day_name)
    data.month_name = data.month_name.map(map_month_name)
    # ==================================================================================================
    '''
        discretizacion horaria
    '''
    def discretizar_horario(df):
        condicion_day = [
            (df.hora >= 0) & (df.hora < 5),   # madrugada
            (df.hora >= 5) & (df.hora < 10),  # mañana
            (df.hora >= 10) & (df.hora < 12),  # media_mañana
            (df.hora >= 12) & (df.hora < 14),  # medio_dia
            (df.hora >= 14) & (df.hora < 16),  # siesta
            (df.hora >= 16) & (df.hora < 20),  # tarde
            (df.hora >= 20) & (df.hora < 24)]  # noche

        resultado_day = ['01_madrugada', '02_mañana', '03_media_mañana',
                         '04_medio_dia', '05_siesta', '06_tarde', '07_noche']

        df['part_of_day'] = np.select(
            condicion_day,
            resultado_day,
            default=-999)
        return df

    '''
        discretizacion de tipo de consumo
    '''
    def discretizar_consumo(df):
        condicion = [
            (df.hora >= 0) & (df.hora < 7),  # cena del dia anterior
            (df.hora >= 7) & (df.hora < 11),  # desayuno
            (df.hora >= 11) & (df.hora < 16),  # almuerzo
            (df.hora >= 16) & (df.hora < 20),  # merienda
            (df.hora >= 20) & (df.hora < 24)]  # cena

        resultado = ['05_cena_ayer', '01_desayuno',
                     '02_almuerzo', '03_merienda', '04_cena']

        df['type_cons'] = np.select(
            condicion,
            resultado,
            default=-999)

        return df

    data = discretizar_horario(data)
    data = discretizar_consumo(data)

    # ==================================================================================================

    condicion_quincena = [
        (data.day <= 15),
        (data.day > 15)
    ]

    resultado_q = ['01_quincena_1', '02_quincena_2']
    # ------------------------------------------------
    condicion_tercio_mes = [
        (data.day <= 10),
        (data.day > 10) & (data.day <= 20),
        (data.day > 20)
    ]

    resultado_t = ['tercio_1', 'tercio_2', 'tercio_3']
    # ------------------------------------------------
    condicion_semana_mes = [
        (data.day <= 7),
        (data.day > 7) & (data.day <= 14),
        (data.day > 14) & (data.day <= 21),
        (data.day > 21) & (data.day <= 28),
        (data.day > 28)
    ]

    resultado_s = ['sem_1', 'sem_2', 'sem_3', 'sem_4', 'sem_5']
    # ------------------------------------------------
    data['quincena'] = np.select(condicion_quincena, resultado_q, default=-999)
    data['tercio_mes'] = np.select(
        condicion_tercio_mes, resultado_t, default=-999)
    data['semana_mes'] = np.select(
        condicion_semana_mes, resultado_s, default=-999)

#     columns_encabezado = ['total_agg','cant_mesas','cant_pedidos',
#                           'year','mes','day','hora','minutos','day_name','month_name','part_of_day',
#                           'quincena','semana_mes','tercio_mes']
#     columns_order = columns_encabezado + [col for col in data.columns if col not in columns_encabezado]
#     data = data[columns_encabezado]

    return data


def facturacion_detalle(df):
    df_1 = (
        df
        .groupby(['date', 'numero', 'denominacion'])
        .agg({'cantidad': sum})
        .rename(columns={'cantidad': 'pedidos'})
    )
    df_1 = df_1[['pedidos']].unstack(level=2)
    df_1.columns = df_1.columns.droplevel(0)
    df_1.columns.name = ''
    df_1['cant_items'] = df_1.apply(lambda x: x.sum(), axis=1)
    df_1['cant_pedidos'] = df_1.count(axis=1, numeric_only=True)

    # -----------------------------------------------------------------

    df_2 = (
        df
        .groupby(['date', 'numero', 'denominacion'])
        .agg({'precio_total_con_iva': sum})
        .rename(columns={'precio_total_con_iva': 'facturado'})
    )
    df_2 = df_2[['facturado']].unstack(level=2)
    df_2.columns = df_2.columns.droplevel(0)
    df_2.columns.name = ''
    df_2 = df_2.fillna(0)
    df_2['facturado'] = df_2.apply(lambda x: x.sum(), axis=1)

    # -----------------------------------------------------------------

    data = pd.merge(
        left=df_1[['cant_items', 'cant_pedidos']].astype('int'),
        right=df_2[['facturado']].astype('float'),
        on=['date', 'numero'],
        how='outer'
    )

    # -----------------------------------------------------------------

    data = data.reset_index(level=1)
    data['cant_mesas'] = data.numero.apply(lambda x: 1 if x != '' else 0)

    data.rename(columns={'numero': 'comprobante'}, inplace=True)

    data = data.resample('H').agg({'facturado': sum,
                                   'cant_items': sum,
                                   'cant_pedidos': sum,
                                   'cant_mesas': sum})

    data = (
        data
        .assign(fecha=lambda df_: df_.index.date)
        .assign(year=lambda df_: df_.index.year)
        .assign(mes=lambda df_: df_.index.month)
        .assign(day=lambda df_: df_.index.day)
        .assign(day_name=lambda df_: df_.index.day_name())
        .assign(month_name=lambda df_: df_.index.month_name())
        .assign(hora=lambda df_: df_.index.hour)
    )
    # ==================================================================================================
    '''
        recodificar month_name y day_name
    '''
    map_day_name = {
        'Monday': '01_Monday',
        'Tuesday': '02_Tuesday',
        'Wednesday': '03_Wednesday',
        'Thursday': '04_Thursday',
        'Friday': '05_Friday',
        'Saturday': '06_Saturday',
        'Sunday': '07_Sunday'
    }

    map_month_name = {
        'January': '01_January',
        'February': '02_February',
        'March': '03_March',
        'April': '04_April',
        'May': '05_May',
        'June': '06_June',
        'July': '07_July',
        'August': '08_August',
        'September': '09_September',
        'October': '10_October',
        'November': '11_November',
        'December': '12_December'
    }
    data.day_name = data.day_name.map(map_day_name)
    data.month_name = data.month_name.map(map_month_name)
    # ==================================================================================================
    '''
        discretizacion horaria
    '''
    def discretizar_horario(df):
        condicion_day = [
            (df.hora >= 0) & (df.hora < 5),   # madrugada
            (df.hora >= 5) & (df.hora < 10),  # mañana
            (df.hora >= 10) & (df.hora < 12),  # media_mañana
            (df.hora >= 12) & (df.hora < 14),  # medio_dia
            (df.hora >= 14) & (df.hora < 16),  # siesta
            (df.hora >= 16) & (df.hora < 20),  # tarde
            (df.hora >= 20) & (df.hora < 24)]  # noche

        resultado_day = ['01_madrugada', '02_mañana', '03_media_mañana',
                         '04_medio_dia', '05_siesta', '06_tarde', '07_noche']

        df['part_of_day'] = np.select(
            condicion_day,
            resultado_day,
            default=-999)
        return df

    '''
        discretizacion de tipo de consumo
    '''
    def discretizar_consumo(df):
        condicion = [
            (df.hora >= 0) & (df.hora < 7),  # cena del dia anterior
            (df.hora >= 7) & (df.hora < 11),  # desayuno
            (df.hora >= 11) & (df.hora < 16),  # almuerzo
            (df.hora >= 16) & (df.hora < 20),  # merienda
            (df.hora >= 20) & (df.hora < 24)]  # cena

        resultado = ['05_cena_ayer', '01_desayuno',
                     '02_almuerzo', '03_merienda', '04_cena']

        df['type_cons'] = np.select(
            condicion,
            resultado,
            default=-999)

        return df

    data = discretizar_horario(data)
    data = discretizar_consumo(data)

    # ==================================================================================================

    condicion_quincena = [
        (data.day <= 15),
        (data.day > 15)
    ]

    resultado_q = ['01_quincena_1', '02_quincena_2']
    # ------------------------------------------------
    condicion_tercio_mes = [
        (data.day <= 10),
        (data.day > 10) & (data.day <= 20),
        (data.day > 20)
    ]

    resultado_t = ['tercio_1', 'tercio_2', 'tercio_3']
    # ------------------------------------------------
    condicion_semana_mes = [
        (data.day <= 7),
        (data.day > 7) & (data.day <= 14),
        (data.day > 14) & (data.day <= 21),
        (data.day > 21) & (data.day <= 28),
        (data.day > 28)
    ]

    resultado_s = ['sem_1', 'sem_2', 'sem_3', 'sem_4', 'sem_5']
    # ------------------------------------------------
    data['quincena'] = np.select(condicion_quincena, resultado_q, default=-999)
    data['tercio_mes'] = np.select(
        condicion_tercio_mes, resultado_t, default=-999)
    data['semana_mes'] = np.select(
        condicion_semana_mes, resultado_s, default=-999)

    return data


def cambiar_fecha(registro):
    registro.fecha = registro.fecha-timedelta(days=1)
    registro.day = registro.fecha.day
    registro.mes = registro.fecha.month
    registro.type_cons = '04_cena'
    registro.hora = 23
    return registro


def consumo_detalle(df):
    data = df.reset_index().groupby(['date', 'denominacion']).agg(
        {'cantidad': sum}).unstack(level=1).fillna(0)
    data.columns = data.columns.droplevel(0)
    data.columns.name = ''

    data = data.resample('H').agg({col: sum for col in data.columns})
    return data


def graficar_serie_temporal(df, var_analisis=['facturado']):

    df = df.set_index('date')

    df = df[['day_name', 'semana_mes', 'facturado'] + var_analisis]

    traces = []
    colors = colors = ["rebeccapurple",
                       "paleturquoise", 'green', 'orange', "purple"]

    for i, var in enumerate(var_analisis):
        traces.append(go.Scatter(
            x=df.index,
            y=df[var],
            line={'color': colors[i],
                  'width': 1,
                  #                               'dash' : 'dash'
                  },
            mode='lines+markers',
            name=f'{var}',
            marker={
                'color': colors[i],
                'size': 4}),
        )

    traces.append(go.Scatter(
        x=df.index,
        y=df.facturado,
        line={'color': 'red',
              'width': 1},
        mode='lines+markers',
        name='facturado',
        marker={
            'color': 'red',
            'size': 4},
        yaxis='y2'
    ))

    fig = go.Figure({
        'data': traces,
        'layout': go.Layout(
            autosize=True,
            hovermode="x",
            height=600,
            width=1900,
            title={
                    'text': "Analisis de Linea de tiempo",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20,
                             'color': 'black'}
                    },
            template='plotly_dark',
            font={'size': 10},
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        # dict(count=1,
                        #          label="1h",
                        #          step="hour",
                        #          stepmode="backward"),
                        # dict(count=3,
                        #      label="3h",
                        #      step="hour",
                        #      stepmode="backward"),
                        # dict(count=12,
                        #      label="12h",
                        #      step="hour",
                        #      stepmode="backward"),
                        dict(count=1,
                             label="1d",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="7d",
                             step="day",
                             stepmode="backward"),
                        dict(count=14,
                             label="14d",
                             step="day",
                             stepmode="backward")
                    ]),
                    font=dict(color='black')
                ),
                rangeslider=dict(
                    visible=True,
                ),
                type="date",
            )
        )
    })

    fig.update_layout(
        #         title = 'Time Series with Custom Date-Time Format',
        xaxis_tickformat='(%a) %d %b %y<br>%H:%M',
        legend=dict(
            x=0.99,
            xanchor='right',
            y=0.99,
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            font={'size': 14}
        ),
        yaxis=dict(
            title=dict(text="Total pedidos y mesas"),
            side="left",
            range=[0, 250],
        ),
        yaxis2=dict(
            title=dict(text="Total Facturado"),
            side="right",
            range=[0, df.facturado.max()],
            overlaying="y",
            tickmode="sync",
        ),
    )

    return fig


def graficar_pct_change(df, var_analisis='cant_mesas', var_agg='type_cons'):

    titulos_1 = {
        'cant_mesas': 'cantidad de mesas por',
        'cant_pedidos': 'cantidad de pedidos por',
        'facturado': 'facturacion por',
        'pedidos_promedio': 'pedidos promedio por',
        'importe_prom_pedido': 'facturacion promedio por pedido por',
        'importe_prom_mesa': 'facturacion promedio por mesa por'
    }

    titulos_2 = {
        'type_cons': 'tipo de consumo',
        'part_of_day': 'segmento del dia',
        'day_name': 'dia de la semana',
        'quincena': 'quincena',
        'tercio_mes': 'tercio del mes',
        'semana_mes': 'semana del mes',
    }

    df = (
        df
        .groupby(['month_name', var_agg])
        .agg({'facturado': sum, 'cant_pedidos': sum, 'cant_mesas': sum})
        .assign(pedidos_promedio=lambda df_: df_.cant_pedidos/df_.cant_mesas)
        .assign(importe_prom_pedido=lambda df_: df_.facturado/df_.cant_pedidos)
        .assign(importe_prom_mesa=lambda df_: df_.facturado/df_.cant_mesas)
    )[[var_analisis]].unstack(level=1).pct_change()

    df.columns = df.columns.droplevel(0)
    df.columns.name = ''
    df.index.name = ''

    colors = ["red", 'orange', "blue", 'green',
              "yellow", "purple", "lightblue"]

    traces = [
        go.Bar(x=df.index,
               y=np.round(df[col]*100, 2),
               name=str(col),
               marker_color=colors[i],
               hovertemplate="<br>".join([
                   "%{y}%"]),
               orientation='v') for i, col in enumerate(df.columns)
    ]

    fig = go.Figure({
        'data': traces,
        'layout': go.Layout(
                    autosize=True,
                    #                     plot_bgcolor    = PLOT_BGCOLOR,
                    #                     paper_bgcolor   = PAPER_BGCOLOR,
                    hovermode="x unified",
                    height=600,
                    width=950,
                    title={
                        'text': f"Variacion de {titulos_1[var_analisis]} <br> {titulos_2[var_agg]}",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 20,
                                 'color': 'white'}
                    },
                    template='plotly_dark',
                    font={'size': 10},
                    )
    })
    fig.update_layout(
        #         title='Variacion de cantidad de mesas por mes',
        #         xaxis_tickfont_size=14,
        xaxis=dict(
            title='Pct Change (%)',
            titlefont_size=15,
            tickfont_size=10,
        ),
        legend=dict(
            x=0,
            y=1.05,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            font={'size': 10},
            orientation="h",
        )
    )
    return fig


def graficar_hm_facturacion(df, rows, cols):

    df = pd.pivot_table(df,
                        index=rows,
                        columns=cols,
                        # values='cant_pedidos',
                        values='facturado',
                        aggfunc=lambda x: np.floor(np.sum(x))).fillna(0)  # .reset_index()

    # 'RdBu_r', 'inferno', 'Bluered_r', 'px.colors.cyclical.IceFire'
    # scale_color = ["red", 'orange', "yellow", 'lightgreen', "green"]
    scale_color = ["green",
                   'lightgreen', 'skyblue', "yellow", 'orange', "red"]
    # scale_color = 'icefire'

    fig = px.imshow(df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=scale_color,
                    origin='upper')

    fig.update_layout(
        template='plotly_dark',
        height=600)

    return fig


def group_by(df, qual_val, quant_val, name_columns):
    df = (
        df
        .groupby(qual_val)
        .agg({
            v: lambda x: {
                #                 'total': np.round(sum(x),2),
                'promedio': np.round(np.mean(x), 2),
                'per50': np.round(np.median(x), 2)
            } for v in quant_val
        })
    )
    df.columns = name_columns
    df['info'] = df.apply(lambda x: str(
        {key: val for key, val in x.items()}), axis=1)

    return df[['info']]


def build_hierarchical_dataframe(df, levels, value_column, color_columns=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(
        columns=['id', 'parent', 'value', 'color', 'info'])

    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(
            columns=['id', 'parent', 'value', 'color', 'info'])
        dfg = pd.concat([
            df.groupby(levels[i:]).sum(),
            group_by(df=df,
                     qual_val=levels[i:],
                     #  quant_val=['facturado', 'cant_mesas', 'cant_pedidos'],
                     #  name_columns=['facturado', 'cant_mesas', 'cant_pedidos']
                     quant_val=['precio_lista'],
                     name_columns=['precio_lista']
                     )
        ], axis=1)

        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        df_tree['info'] = dfg['info']
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='',
                              value=df[value_column].sum(),
                              color=df[color_columns[0]].sum() / df[color_columns[1]].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees


def graficar_treemap_facturacion(df):

    levels = ['type_cons', 'day']
    # color_columns = ['facturado', 'cant_mesas']
    color_columns = ['facturado']
    value_column = 'facturado'

    scale_green = ['green' for i in range(0, 4, 1)]
    scale_yellow = ['yellow' for i in range(0, 2, 1)]
    scale_orange = ['orange' for i in range(0, 1, 1)]
    scale_red = ['red' for i in range(0, 4, 1)]
    color_scale = scale_green+scale_yellow+scale_orange+scale_red

    df_all_trees = build_hierarchical_dataframe(
        df, levels, value_column, color_columns)
    # average_score = df['facturado'].sum() / df['cant_mesas'].sum()

#     fig = px.treemap(df_all_trees, path=['parent','id'], values='value',
#                   color='color', hover_data=['info'],
#                   color_continuous_scale='RdBu',
#                   color_continuous_midpoint=average_score)

#     hovertemplate = '<b>%{label} </b> <br> Facturado: %{value}<br> Promedio por pedido: %{color:.2f}'
    # hovertemplate = '<b>Promedio por mesa: $%{color:.2f}'

    traces = []

    traces.append(go.Treemap(
        labels=df_all_trees['id'],
        parents=df_all_trees['parent'],
        values=df_all_trees['value'],
        branchvalues='total',
        textinfo='label+percent entry+value',
        # texttemplate='<b>Facturado: $%{value:.2f} </b><br><b>Porcentaje: %{percent: .2f}%</b>',
        marker=dict(
            colors=df_all_trees['color'],
            colorscale=color_scale,
            # cmid=average_score
        ),
        # hovertemplate=hovertemplate,
        name='',
        maxdepth=2
    ))

    fig = go.Figure({
        'data': traces,
        'layout': go.Layout(
                    autosize=True,
                    hovermode="x unified",
                    height=200,
                    width=1900,
                    title={
                        'text': "Diagrama jerarquico de facturacion",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 20,
                                 'color': 'black'}
                    },
                    template='plotly_dark',
                    font={'size': 14}
                    )
    })

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


def graficar_treemap_consumo(df):

    # levels = ['familia', 'month_name']
    levels = ['familia', 'month_name']
    color_columns = ['precio_lista', 'cantidad']
    value_column = 'precio_lista'

    scale_green = ['green' for i in range(0, 8, 1)]
    scale_yellow = ['yellow' for i in range(0, 4, 1)]
    scale_orange = ['orange' for i in range(0, 1, 1)]
    scale_red = ['red' for i in range(0, 3, 1)]

    # , scale_green+scale_yellow+scale_orange+scale_red
    color_scale = ["blue", "green", "yellow", "red"]

    df_all_trees = build_hierarchical_dataframe(
        df, levels, value_column, color_columns)

    print('df_all_trees')
    print(df_all_trees)

    average_score = df['precio_lista'].sum() / df['cantidad'].sum()

    hovertemplate = '<b>%{label} </b> <br> Facturado: %{value}<br> Promedio por pedido: %{color:.2f}'

    traces = []

    traces.append(go.Treemap(
        labels=df_all_trees['id'],
        parents=df_all_trees['parent'],
        values=df_all_trees['value'],
        branchvalues='total',
        textinfo='label+percent root+percent parent+value',
        texttemplate='<b>%{label}<br> $%{value: 0.2s}</b> <br><br>  <br> %{percentRoot} de %{root} <br> %{percentParent} de %{parent}',
        textposition='middle center',
        marker=dict(
            colors=df_all_trees['color'],
            colorscale=color_scale,
            cmid=average_score,
            showscale=True),
        hovertemplate=hovertemplate,
        name='',
        #         maxdepth=-1
    ))

    fig = go.Figure({
        'data': traces,
        'layout': go.Layout(
                    autosize=True,
                    hovermode="x unified",
                    height=200,
                    # width=1900,
                    #                     title           ={
                    #                         'text': "Diagrama jerarquico de facturacion",
                    #                         'y':0.95,
                    #                         'x':0.5,
                    #                         'xanchor': 'center',
                    #                         'yanchor': 'top',
                    #                         'font' : {'size':20,
                    #                                   'color':'black'}
                    #                     },
                    template='plotly_dark',
                    font={'size': 15}
                    )
    })

    fig.update_layout(margin=dict(t=0, l=0, r=25, b=0))
    return fig


def updates_kpis_data_store(df):

    # -----------------------------------------------------------------------------

    height = 130
    paper_bgcolor = 'rgba(206, 8, 1, 0.8)'  # 'rgba(153, 188, 196, 0.53)'
    font_color = 'white'
    # number = {"font": {"size": 45,
    #                    'color': font_color}}
    # font_size = 12

    best_day_month = (
        df
        .groupby(['month_name', 'day'])
        .agg({
            'facturado': lambda x: pd.Series(x).sum()
        })
        .reset_index()
        .sort_values(by='facturado', ascending=False)
    ).iloc[:5, :].apply(lambda x: f'dia: {x.day}, mes: {x.month_name}, facturado: ${x.facturado:,.0f}', axis=1).values[0]

    best_part_of_day = (
        df
        .groupby(['month_name', 'part_of_day'])
        .agg({
            'facturado': lambda x: pd.Series(x).sum()
        })
        .reset_index()
        .sort_values(by='facturado', ascending=False)
    ).iloc[:5, :].apply(lambda x: f'part_of_day: {x.part_of_day}, facturado: ${x.facturado:,.0f}', axis=1).values[0]

    best_time = (
        df
        .groupby(['month_name', 'hora'])
        .agg({
            'facturado': lambda x: pd.Series(x).sum()
        })
        .reset_index()
        .sort_values(by='facturado', ascending=False)
    ).iloc[:5, :].apply(lambda x: f'hora: {x.hora}, facturado: ${x.facturado:,.0f}', axis=1).values[0]

    # -----------------------------------------------------------------------------

    fig_3 = go.Figure(go.Indicator(
        mode="number",
        value=np.round(df.facturado.sum(), 0),
        title={"text": f"TOTAL FACTURADO MENSUAL", 'font': {
            'size': 20}},
        number={"font": {"size": 45,
                         'color': font_color}}
    ))

    fig_3.update_layout({
        "margin": {"l": 0, "r": 0, "b": 0, "t": 20},
        "autosize": True,
        'height': height,
        'paper_bgcolor': paper_bgcolor,
        'font': {'color': font_color}
    })

    return html.Div(id='dashboard-kpis',
                    children=[
                        dbc.Row([
                            dbc.CardGroup([
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-day-of-month',
                                        children=[
                                            html.H5("Mejor dia del mes"),
                                            html.H5(f"{best_day_month}")
                                        ],
                                        style={'padding-top': '20px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'auto', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-part-of-day-of-month',
                                        children=[
                                            html.H5("Mejor franja"),
                                            html.H5(f"{best_part_of_day}")
                                        ],
                                        style={'padding-top': '20px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='kpi-fact-total',
                                        children=[
                                            dcc.Graph(
                                                id='graph_kpi_fact_total', figure=fig_3
                                            )
                                        ]
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-ticket',
                                        children=[
                                            html.H5("Mejor hora"),
                                            html.H5(f"{best_time}")
                                        ],
                                        style={'padding-top': '20px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ])
                            ])
                        ])
                    ])


def updates_kpis_raw_df(df):

    height = 250
    paper_bgcolor = 'rgba(206, 8, 1, 0.8)'  # 'rgba(206, 8, 1, 0.8)'
    font_color = 'white'
    top_item = 5

    def make_data_to_graf(df, var_analisis='familia', top_item=5):
        thr = min(top_item, df[var_analisis].nunique())
        return (
            df
            .groupby(['month_name', var_analisis])
            .agg({
                'precio_lista': lambda x: pd.Series(x).sum(),
                'precio_costo': lambda x: pd.Series(x).sum(),
                'rentabilidad': lambda x: pd.Series(x).sum(),
                'denominacion': lambda x: pd.Series(x).size
            })
            .rename(columns={'precio_lista': 'facturado', 'precio_costo': 'costos', 'denominacion': 'cant_items_familia'})
            .reset_index()
            .assign(avg_renta=lambda df_: df_.rentabilidad/df_.cant_items_familia)
            .sort_values(by='facturado', ascending=False)
        ).iloc[:thr, :].sort_values(by='facturado', ascending=True)

    def make_graf_top_items(df, x_var, y_var):
        paper_bgcolor = 'rgba(206, 8, 1, 0)'  # 'rgba(153, 188, 196, 0.53)'
        font_color = 'white'
        fontsize = 10

        traces = [
            go.Bar(
                x=df[df.month_name == month][x_var],
                y=df[df.month_name == month][y_var],
                name=month,
                # marker_color='white',
                hovertemplate="<br>".join([
                    "%{x:,.0f}$"]),
                orientation='h') for month in df.month_name.unique()
        ]

        fig = go.Figure({
            'data': traces,
            'layout': go.Layout(
                        autosize=True,
                        hovermode="closest",
                        font={'size': fontsize},
                        )
        })
        fig.update_layout({
            'autosize': True,
            'height': 160,
            'barmode': 'relative',
            'margin': dict(l=50, r=10, t=10, b=0),
            'paper_bgcolor': paper_bgcolor,
            'plot_bgcolor': paper_bgcolor,
            'font': {'color': font_color},
            'hoverlabel': {
                'bgcolor': "rgba(206, 8, 1, 0.8)",
                'font_size': 12,
                'font_color': 'white'
            },
            'legend': {
                'orientation': "h",
                'yanchor': "bottom",
                'y': 0.95,
                'xanchor': "left",
                'x': 0
            }
        })

        return fig

    best_family = make_data_to_graf(df=df,
                                    var_analisis='familia',
                                    top_item=5)

    best_type_cons = make_data_to_graf(df=df,
                                       var_analisis='type_cons',
                                       top_item=5)

    best_zona_denom = make_data_to_graf(df=df,
                                        var_analisis='zona_denom',
                                        top_item=5)

    best_day_name = make_data_to_graf(df=df,
                                      var_analisis='day_name',
                                      top_item=5)

    best_product = make_data_to_graf(df=df,
                                     var_analisis='denominacion',
                                     top_item=5)

    print('best_family', best_family)

    return html.Div(id='dashboard-kpis-raw-df',
                    children=[
                        dbc.Row([
                            dbc.CardGroup([
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-family',
                                        children=[
                                            html.H6("Mejor rubro facturacion"),
                                            html.H6(
                                                f"{best_family.apply(lambda x: f'familia: {x.familia}, facturado: ${x.facturado:,.0f}', axis=1).values[-1]}"),
                                            dcc.Graph(id='best_family', figure=make_graf_top_items(
                                                df=best_family, x_var='facturado', y_var='familia'))
                                        ],
                                        style={'padding-top': '20px', 'padding-left': '5px', 'padding-right': '5px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'auto', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-best-type-cons',
                                        children=[
                                            html.H6("Mejor tipo de consumo"),
                                            html.H6(
                                                f"{best_type_cons.apply(lambda x: f'type_cons: {x.type_cons}, facturado: ${x.facturado:,.0f}', axis=1).values[-1]}"),
                                            dcc.Graph(id='best_type_cons', figure=make_graf_top_items(
                                                df=best_type_cons, x_var='facturado', y_var='type_cons'))
                                        ],
                                        style={'padding-top': '20px', 'padding-left': '5px', 'padding-right': '5px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-zona-denom',
                                        children=[
                                            html.H6("Mejor zona"),
                                            html.H6(
                                                f"{best_zona_denom.apply(lambda x: f'zona: {x.zona_denom}, facturado: ${x.facturado:,.0f}', axis=1).values[-1]}"),
                                            dcc.Graph(id='best_zona_denom', figure=make_graf_top_items(
                                                df=best_zona_denom, x_var='facturado', y_var='zona_denom'))
                                        ],
                                        style={'padding-top': '20px', 'padding-left': '5px', 'padding-right': '5px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-day_name',
                                        children=[
                                            html.H6("Mejor dia de la semana"),
                                            html.H6(
                                                f"{best_day_name.apply(lambda x: f'dia: {x.day_name}, facturado: ${x.facturado:,.0f}', axis=1).values[-1]}"),
                                            dcc.Graph(id='best_day_name', figure=make_graf_top_items(
                                                df=best_day_name, x_var='facturado', y_var='day_name'))
                                        ],
                                        style={'padding-top': '20px', 'padding-left': '5px', 'padding-right': '5px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ]),
                                dbc.Card([
                                    html.Div(
                                        id='div-kpi-best-product',
                                        children=[
                                            html.H6("Producto mas vendido"),
                                            html.H6(
                                                f"{best_product.apply(lambda x: f'producto: {x.denominacion}, facturado: ${x.facturado:,.0f}', axis=1).values[-1]}"),
                                            dcc.Graph(id='best_day_name', figure=make_graf_top_items(
                                                df=best_product, x_var='facturado', y_var='denominacion'))
                                        ],
                                        style={'padding-top': '20px', 'background': paper_bgcolor, 'height': height,
                                               'text-align': 'center', 'text-justify': 'center', 'justify': 'center'}
                                    ),
                                ]),
                            ])
                        ])
                    ])


def make_cal_plot(df):

    df = df.set_index('date').asfreq('H').resample('D').agg(
        {'facturado': sum}).reset_index()
    color = ["green", 'lightgreen', "yellow", 'orange', "red"]

    f = calplot(
        df,
        x='date',
        y='facturado',
        gap=1,
        colorscale=color,
        month_lines_width=3,
        month_lines_color="black",
        name='facturado',
        showscale=True,
        dark_theme=True
    )

    f.update_layout(
        #         plot_bgcolor    = plot_bgcolor,
        #         paper_bgcolor   = paper_bgcolor,
        # template='plotly_dark',
        height=200,
        hovermode='closest',  # "x unified",
        title={
            'text': f'Mapa de calor de facturacion por dia de la semana',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 20,
                'color': 'black'}
        },
        font={'size': 12, 'color': 'black'}
    )

    return f


defaultColDef = {
    "filter": True,
    "floatingFilter": True,
    "resizable": True,
    "sortable": True,
    "editable": True,
    "minWidth": 100,
}

external_stylesheets = [dbc.themes.CYBORG]

app = Dash()

server = app.server

app.layout = html.Div([
    dcc.Store(id='memory', storage_type='memory'),
    dcc.Store(id='memory-raw-df', storage_type='memory'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='div-button-update', children=[
        dbc.CardGroup([
            dbc.Card([
                dbc.Button('Serie Temporal',
                           id='button-ts', n_clicks=0)]),
            dbc.Card([
                dbc.Button('HeatMap',
                           id='button-heatmap', n_clicks=0),]),
            dbc.Card([
                dbc.Button('Variacion',
                           id='button-change', n_clicks=0),]),
            dbc.Card([
                dbc.Button('Consumo jerarquico',
                           id='button-consumo', n_clicks=0),])
        ])
    ]),
    html.Br(),
    html.Div(id='kpi-data-store', children=[]),
    html.Div(id='kpi-raw-df', children=[]),
    html.Div(id='div-graf-ts', children=[]),
    html.Div(id='div-tree-map-consumo-dia', children=[]),
    html.Div(id='div-heat-map', children=[]),
    html.Div(id='div-graf-change', children=[]),
    html.Div(id='div-graf-facturacion', children=[]),
    html.Div(id='div-hierarchical', children=[]),
    html.Br(),
    html.Div(id='output-div'),
    html.Div(id='load-datatable',
             children=[
                 dcc.Loading(children=[html.Div(id="output-datatable")],
                             color="#119DFF", type="dot", fullscreen=False,)
             ]),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded), header=2)
    except Exception as e:
        print(e)
        return html.Div([
            'Hubo un error de procesamiento de archivo'
        ])
    return df


@app.callback([Output('div-hierarchical', 'children')],
              Input('button-consumo', 'n_clicks'),
              Input('memory', 'data'),
              prevent_inicial_call=True)
def hierarchical_graf(btn, data):
    if btn % 2 == 1:
        df = pd.DataFrame(data)
        return [
            dbc.Row([
                dbc.Row([
                    dbc.Card([
                        dcc.Graph(id='hiera-graf-1',
                                  figure=graficar_treemap_facturacion(df))
                    ])
                ]),
                dbc.Row([
                    dbc.Card([
                        # dcc.Graph(id='hiera-graf-2',
                        #           figure=graficar_treemap_consumo(pd.DataFrame(data2)))
                    ])
                ]),
            ])

        ]
    else:
        return [None]


@app.callback([Output('div-heat-map', 'children')],
              Input('button-heatmap', 'n_clicks'),
              Input('memory', 'data'),
              prevent_initial_call=True)
def heat_map(btn, data):
    if btn % 2 == 1:
        df = pd.DataFrame(data)
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dcc.Graph(id='hm_3', figure=graficar_hm_facturacion(
                                    df, 'day', 'hora'))
                            ]),
                        ], width=6),
                        dbc.Col([
                                dcc.Graph(id='cal-plot',
                                          figure=make_cal_plot(df))
                                ], width=6)
                    ]),
                    # dbc.Row([
                    #     dbc.Card([
                    #         dcc.Graph(id='hm_2', figure=graficar_hm_facturacion(
                    #             df, 'month_name', 'day'))
                    #     ]),
                    # ]),
                    # dbc.Row([
                    #     dbc.Card([
                    #         dcc.Graph(id='hm_1', figure=graficar_hm_facturacion(
                    #             df, 'month_name', 'part_of_day'))
                    #     ])
                    # ]),
                    # dbc.Row([
                    #     dbc.Card([
                    #         dcc.Graph(id='hm_4', figure=graficar_hm_facturacion(
                    #             df, 'semana_mes', 'type_cons'))
                    #     ])
                    # ])
                ])
            ])
        ]
    else:
        return [None]


@app.callback(Output('div-graf-change', 'children'),
              Input('button-change', 'n_clicks'),
              Input('memory', 'data'),
              prevent_initial_call=True)
def pct_change(btn, data):
    if btn % 2 == 1:
        df = pd.DataFrame(data)
        return dbc.Row([
            dbc.Row([
                dbc.CardGroup([
                    dbc.Card([
                        dcc.Graph(id='pct-change-1', figure=graficar_pct_change(
                            df=df, var_analisis='facturado', var_agg='day_name'))
                    ]),
                    dbc.Card([
                        dcc.Graph(id='pct-change-2', figure=graficar_pct_change(
                            df=df, var_analisis='facturado', var_agg='part_of_day'))
                    ])
                ])
            ]),
            dbc.Row([
                dbc.CardGroup([
                    dbc.Card([
                        dcc.Graph(id='pct-change-3', figure=graficar_pct_change(
                            df=df, var_analisis='pedidos_promedio', var_agg='type_cons'))
                    ]),
                    dbc.Card([
                        dcc.Graph(id='pct-change-4', figure=graficar_pct_change(
                            df=df, var_analisis='importe_prom_mesa', var_agg='part_of_day'))
                    ])
                ])
            ])
        ])

    else:
        return None


@app.callback([Output('div-graf-ts', 'children')],
              Input('button-ts', 'n_clicks'),
              Input('memory', 'data'),
              prevent_initial_call=True)
def time_series(btn, data):
    if btn % 2 == 1:
        df = pd.DataFrame(data)
        f = graficar_serie_temporal(
            df=df, var_analisis=['cant_pedidos', 'cant_mesas'])
        return [
            dbc.Card([
                dcc.Graph(id='ts', figure=f)
            ])
        ]
    else:
        return [None]


@app.callback([Output('output-datatable', 'children'),
               Output('memory', 'data'),
               Output('memory-raw-df', 'data'),
               Output('kpi-data-store', 'children'),
               Output('kpi-raw-df', 'children'),
               Output('div-tree-map-consumo-dia', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        dfs = [
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        # ************************************************************************************************
        # VER POLARS PARA ESTA PARTE
        # ************************************************************************************************

        df = armarData(pd.concat(dfs, axis=0))

        df = pd.merge(
            left=df,
            right=familia_denominacion,
            on='denominacion',
            how='left'
        )

        df = insert_calendar_vals(df)

        facturacion = facturacion_detalle(df)

        condition = (facturacion.type_cons == '05_cena_ayer')
        facturacion.loc[condition] = facturacion.loc[condition].apply(
            cambiar_fecha, axis=1)

        consumo = consumo_detalle(df)

        def dic_consumo(registro):
            columnas = consumo.columns
            dic = {f'{col}': registro[col]
                   for col in columnas if registro[col] > 0}
            return dic

        consumo['info_consumo'] = consumo.apply(
            lambda x: dic_consumo(x), axis=1)

        data = pd.merge(
            left=consumo[['info_consumo']],
            right=facturacion,
            on='date',
            how='right'
        ).reset_index()

        data_store = data.copy()

        columns_use_data = data.columns[:5]

        data['tabla_consumo'] = ''

        for i, r in data.iterrows():

            info_consumo = data.at[i, 'info_consumo']

            subset = pd.DataFrame(data=info_consumo.values(),
                                  index=info_consumo.keys(),
                                  columns=['cantidad']).reset_index().rename(columns={'index': 'item'}).sort_values(by='cantidad', ascending=False)

            tabla = go.Figure(data=[go.Table(
                header=dict(values=list(subset.columns),
                            line_color='paleturquoise',
                            fill_color='paleturquoise',
                            font={'size': 10},
                            align='left'),
                cells=dict(values=[subset[col] for col in subset.columns],
                           line_color='lavender',
                           fill_color='lavender',
                           font={'size': 10},
                           align='left'))
            ])
            tabla.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            data.at[i, 'tabla_consumo'] = tabla

        columnDefs = [
            {
                "headerName": f"{col}",
                "field": f"{col}",
            } for col in columns_use_data
        ]

        columnDefs.append({
            "headerName": "Consumo",
            "field": "tabla_consumo",
            "cellRenderer": "DCC_GraphClickData",
            "maxWidth": 500,
            "minWidth": 500,
        })

        table_df = dag.AgGrid(
            id="table_df",
            className="ag-theme-alpine-dark",
            columnDefs=columnDefs,
            rowData=data.to_dict('records'),
            columnSize="sizeToFit",
            defaultColDef=defaultColDef,
            dashGridOptions={"rowHeight": 180},
            style={"height": 750, "width": "100%"}
        )
        print(data_store.info())
        print(data_store.columns)
        print(data_store)
        print(df.info())
        print(df.columns)
        print(df)

        print(df.query('familia=="menu"').denominacion.unique())

        # print(table_df)
        return [
            html.Div([table_df]),
            data_store.to_dict('records'),
            df.to_dict('records'),
            updates_kpis_data_store(data_store),
            updates_kpis_raw_df(df),
            dcc.Graph(id='tree_map_consumo_dia',
                      figure=graficar_treemap_consumo(df))

        ]
    else:
        raise PreventUpdate


if __name__ == '__main__':
    # app.run_server(port=8002, debug=True)
    app.run_server(debug=False)
