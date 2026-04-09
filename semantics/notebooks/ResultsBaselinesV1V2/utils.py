'''Functions to plot data using matplotlib or plotly.
'''
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DAG_EDGES_MAP = {'cancer': 4,
                 'earthquake': 4,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 17,
                 'child': 25,
                 'insurance': 52}

DAG_NODES_MAP = {'cancer': 5,
                 'earthquake': 5,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 11,
                 'child': 20,
                 'insurance': 27}



def process_model_names_and_runtime_v1_data(v1_data, runtime_data):
    v1_data = v1_data.copy()
    runtime_data = runtime_data.copy()
    v1_data['fact_sourcing_elapsed'] = v1_data.apply(
        lambda row: runtime_data[(runtime_data['n_nodes'] == row['n_nodes']) 
                                 & (runtime_data['n_edges'] == row['n_edges']) 
                                 & (runtime_data['seed'] == row['seed'])]['elapsed_time'].values[0],
        axis=1
    )
    v1_data['elapsed'] = (
        v1_data['elapsed_bsaf_creation'] +
        v1_data['elapsed_model_solution'] +
        v1_data['aba_elapsed'] +
        v1_data['ranking_elapsed'] +
        v1_data['fact_sourcing_elapsed']
    )



    v1_data['model_raw'] = v1_data['fact_ranking_method'] + ' ' + v1_data['model_ranking_method']
    v1_data = v1_data[v1_data['model_raw'].isin([
        'original original',
        'refined_indep_facts original',
        'original refined_indep_facts',
        'original arrows_sum', 
        'original arrows_mean'
    ])].copy()
    v1_data['model'] = v1_data['model_raw'].map({
        'original original': 'ABAPC (Original)',
        'refined_indep_facts original': 'V1.1 Refined Fact Ranking',
        'original refined_indep_facts': 'V1.2 Model Selection by Refined Fact Strengths',
        'original arrows_sum': 'V1.3 Model Selection by Arrows Sum',
        'original arrows_mean': 'V1.4 Model Selection by Arrows Mean'
    })

    # BSAF creation and model solution times are not applicable to original ABAPC model
    v1_data.loc[v1_data['model'] == 'ABAPC (Original)', 'elapsed'] = (
        v1_data.loc[v1_data['model'] == 'ABAPC (Original)', 'fact_sourcing_elapsed'] +
        v1_data.loc[v1_data['model'] == 'ABAPC (Original)', 'aba_elapsed'] + 
        v1_data.loc[v1_data['model'] == 'ABAPC (Original)', 'ranking_elapsed']
    )
    return v1_data


def process_mean_std_sid_data(df, groupby_cols = ['dataset', 'n_nodes', 'n_edges', 'model']):
    df_grouped = df.groupby(groupby_cols, as_index=False).aggregate(
        sid_low_mean=('sid_low', 'mean'),
        sid_high_mean=('sid_high', 'mean'),
        sid_low_std=('sid_low', 'std'),
        sid_high_std=('sid_high', 'std'),
        precision_mean=('precision', 'mean'),
        precision_std=('precision', 'std'),
        recall_mean=('recall', 'mean'),
        recall_std=('recall', 'std'),
        f1_mean=('F1', 'mean'),
        f1_std=('F1', 'std'),
        shd_mean=('shd', 'mean'),
        shd_std=('shd', 'std'),
        nnz_mean=('nnz', 'mean'),
        nnz_std=('nnz', 'std'),
    )
    df_grouped['n_sid_low_mean'] = df_grouped['sid_low_mean'] / df_grouped['n_edges']
    df_grouped['n_sid_high_mean'] = df_grouped['sid_high_mean'] / df_grouped['n_edges']
    df_grouped['n_sid_low_std'] = df_grouped['sid_low_std'] / df_grouped['n_edges']
    df_grouped['n_sid_high_std'] = df_grouped['sid_high_std'] / df_grouped['n_edges']

    df_grouped['n_shd_mean'] = df_grouped['shd_mean'] / df_grouped['n_edges']
    df_grouped['n_shd_std'] = df_grouped['shd_std'] / df_grouped['n_edges']

    return df_grouped


def plot_runtime(df, 
                 names_dict, 
                 colors_dict, 
                 symbols_dict,
                 methods, 
                 plot_width=750, plot_height=300, font_size=20, save_figs=False, output_name="random_graphs_runtime.html"):

    fig = make_subplots(rows=1, cols=1, shared_yaxes=True)
   
    for method in methods:
        method_df = df[df['model'] == method]
        fig.add_trace(
            go.Scatter(
                x=method_df['n_nodes'].astype(str),
                y=method_df['elapsed_mean'],
                error_y=dict(type='data', array=method_df['elapsed_std'], thickness=2),
                mode='lines+markers',
                name=names_dict[method],
                line=dict(color=colors_dict[method], width=2),
                marker=dict(symbol=symbols_dict[method], size=8, color=colors_dict[method]),
                opacity=0.8,
            )
        )

    # Log scale for y-axis
    fig.update_yaxes(type="log", title='log(elapsed time [s])')

    # X axis title
    fig.update_xaxes(title='Number of Nodes (|V|)')

    # Layout and style
    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=1.05),
        template='plotly_white',
        width=plot_width,
        height=plot_height,
        margin=dict(l=10, r=10, b=80, t=10),
        font=dict(size=font_size, family="Serif", color="black")
    )

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html', '.jpeg'), scale=2)

    fig.show()
    return fig



def double_bar_chart_plotly(all_sum, names_dict, colors_dict, 
                            vars_to_plot=['n_sid_low', 'n_sid_high'],
                            names=['Best', 'Worst'],
                            labels=['Normalised SID', ''],
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'ABAPC (Existing)', 'ABAPC (ASPforABA)'],
                            range_y1=[0, 1], range_y2=[0, 1], font_size=20,
                            save_figs=False, output_name="bar_chart.html", debug=False,
                            dist_between_lines=0.1565,
                            intra_dis=0.112,
                            inter_dis=0.137,
                            lin_space=9,
                            nl_space=9,
                            start_pos = 0.039,
                            width=1600,
                            height=700,
                            annot_y=1.015):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for n, var_to_plot in enumerate(vars_to_plot):
        for m, method in enumerate(methods):
            trace_name = method
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                yaxis=f"y{n+1}",
                                offsetgroup=m+len(methods)*n+(1*n),
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=names_dict[trace_name],
                                marker_color=colors_dict[method],
                                opacity=0.6,
                                #  width=0.1
                                showlegend=n==0
                                ))
        if n==0:
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                    y=np.zeros(len(all_sum[(all_sum.model==method)]['dataset'])), 
                                    name='',
                                    offsetgroup=m+1,
                                    marker_color='white',
                                    opacity=1,
                                    # width=0.1
                                    showlegend=False
                                    )
                                    )
    second_ticks = False if len(labels[1]) < 1 else True
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.15, # gap between bars of the same location coordinate.)

            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
            template='plotly_white',
            # autosize=True,
            width=width, 
            height=height,
            margin=dict(
                l=40,
                r=00,
                b=70,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=font_size, family="Serif", color="black"),
            yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=second_ticks, zeroline=True),
            )

    
    for n, (var_to_plot, label, y_range) in enumerate(zip(vars_to_plot, labels, [range_y1, range_y2])):
        fig.update_yaxes(title={'text':f'{label}','font':{'size':font_size}}, secondary_y=n==1, range=y_range)
        if second_ticks == False:
            fig.update_yaxes(title={'text':'','font':{'size':font_size}}, secondary_y=True, range=y_range, showticklabels=False)
    

    name1, name2 = names
    

    n_x_cat = len(all_sum.dataset.unique())
    list_of_pos = []
    left=start_pos
    for i in range(n_x_cat):
        right = left+intra_dis
        list_of_pos.append((left, right))
        left = right+inter_dis

    for s1,s2 in list_of_pos:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s1,
            y=annot_y,
                    text=f"{' '*lin_space}{name1}{' '*(lin_space)}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="left",
            x=s2,
            y=annot_y,
                    text=f"{' '*(nl_space)}{name2}{' '*nl_space}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=font_size,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )
    
    # Add vertical lines between bar groups
    s1 = 0
    sign = -1
    for i in range(n_x_cat*2):  # skip last one
        sign *= -1
        s1 += dist_between_lines
        fig.add_shape(
            type="line",
            x0=s1, x1=s1,
            y0=0, y1=1,
            xref="paper",
            yref="paper",
            line=dict(
                color="grey" if sign>0 else "black",
                width=1,
                dash="dash" if sign>0 else "solid",
            ),
            layer="below"
        )
        


    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))

    fig.show()
    return fig
