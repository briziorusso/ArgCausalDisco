from plotly import graph_objects as go
from plotly.subplots import make_subplots
# import kaleido
import numpy as np
import pandas as pd
##Colors
main_gray = '#262626'
sec_gray = '#595959'
main_blue = '#005383'
sec_blue = '#0085CA'
main_green = '#379f9f' 
sec_green = '#196363' 
main_purple='#9454c4'
sec_purple='#441469'
main_orange='#8a4500'
sec_orange='#b85c00'

def bar_chart_plotly(all_sum, var_to_plot, names_dict, colors_dict, methods, font_size=20, save_figs=False, output_name="bar_chart.html", debug=False):
    fig = go.Figure()
    for method in methods:
        trace_name = 'True Graph Size' if var_to_plot == 'nnz' and method == 'Random' else method
        if 'log' in var_to_plot:
            metric_name = var_to_plot.replace('log_', '')
            display_name = "log(Elapsed Time)" if "lapsed" in metric_name else f"log({metric_name})"
            trace_name = f"Log {trace_name}"
            fig.add_trace(go.Bar(
                x=all_sum[(all_sum.model == method)]['dataset'],
                y=all_sum[(all_sum.model == method)][metric_name + '_mean'],
                error_y=dict(type='data', array=all_sum[(all_sum.model == method)][metric_name + '_std'], visible=True),
                name=trace_name,
                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                opacity=0.6,
            ))
            fig.update_yaxes(type='log')
        else:
            fig.add_trace(go.Bar(
                x=all_sum[(all_sum.model == method)]['dataset'],
                y=all_sum[(all_sum.model == method)][var_to_plot + '_mean'],
                error_y=dict(type='data', array=all_sum[(all_sum.model == method)][var_to_plot + '_std'], visible=True),
                name=trace_name,
                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                opacity=0.6,
            ))

    fig.update_layout(
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=1),
        template='plotly_white',
        width=1600,
        height=700,
        margin=dict(l=40, r=40, b=80, t=20),
        hovermode='x unified',
        font=dict(size=font_size, family='Serif', color='black'),
    )

    # fig.add_annotation(
    #     xref='paper',
    #     yref='paper',
    #     xanchor='center',
    #     x=0,
    #     yanchor='bottom',
    #     y=-0.05,
    #     text='Dataset:',
    #     showarrow=False,
    #     font=dict(family='Serif', size=font_size, color='Black'),
    # )

    if 'n_' in var_to_plot or 'p_' in var_to_plot:
        orig_y = var_to_plot.replace('n_', '').replace('p_', '').upper()
        fig.update_yaxes(title={'text': f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG', 'font': {'size': font_size}})
    elif var_to_plot == 'nnz':
        fig.update_yaxes(title={'text': 'Number of Edges in DAG', 'font': {'size': font_size}})
    else:
        fig.update_yaxes(title={'text': var_to_plot.title(), 'font': {'size': font_size}})

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html', '.jpeg'))

    fig.show()



def double_bar_chart_plotly(all_sum, vars_to_plot, names_dict, colors_dict, 
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)'],
                            range_y1=None, range_y2=None, font_size=20,
                            save_figs=False, output_name="bar_chart.html", rect_exp=0.02, debug=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    second_ticks = not all('SID' in var for var in vars_to_plot)

    for n, var_to_plot in enumerate(vars_to_plot):
        for m, method in enumerate(methods):
            trace_name = 'True Graph Size' if var_to_plot == 'nnz' and method == 'Random' else method
            fig.add_trace(go.Bar(
                x=all_sum[(all_sum.model == method)]['dataset'],
                yaxis=f"y{n+1}",
                offsetgroup=m + len(methods) * n + (1 * n),
                y=all_sum[(all_sum.model == method)][var_to_plot + '_mean'],
                error_y=dict(type='data', array=all_sum[(all_sum.model == method)][var_to_plot + '_std'], visible=True),
                name=trace_name,
                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                opacity=0.6,
                showlegend=n == 0
            ))
        if n == 0:
            fig.add_trace(go.Bar(
                x=all_sum[(all_sum.model == method)]['dataset'],
                y=np.zeros(len(all_sum[(all_sum.model == method)]['dataset'])),
                name='',
                offsetgroup=m + 1,
                marker_color='white',
                opacity=1,
                showlegend=False
            ))

    legend_y = 1.23
    top_margin = max(80, int(font_size * 2.8))
    fig.update_layout(
        barmode='group',
        bargap=0.08,
        bargroupgap=0.05,
        legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=legend_y),
        template='plotly_white',
        width=1600,
        height=700,
        margin=dict(l=40, r=40, b=70, t=top_margin),
        hovermode='x unified',
        font=dict(size=font_size, family='Serif', color='black'),
        yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=second_ticks, zeroline=True),
    )

    # fig.update_traces(width=0.36, selector=dict(type='bar'))

    # fig.add_annotation(
    #     xref='paper',
    #     yref='paper',
    #     xanchor='center',
    #     x=0,
    #     yanchor='bottom',
    #     y=-0.08,
    #     text='Dataset:',
    #     showarrow=False,
    #     font=dict(family='Serif', size=font_size, color='Black'),
    # )

    for n, var_to_plot in enumerate(vars_to_plot):
        if vars_to_plot == ['precision', 'recall']:
            range_y = range_y1 if range_y1 is not None else [0, 1.3]
        elif vars_to_plot == ['fdr', 'tpr']:
            range_y = [0, 1]
        elif 'shd' in var_to_plot or 'SID' in var_to_plot:
            if range_y1 is None and range_y2 is None:
                if 'high' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_high_mean']) + 0.3]
                elif 'low' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_low_mean']) + 0.3]
                else:
                    range_y = [0, 2] if n == 0 else [0, max(all_sum['p_SID_mean']) + 0.3]
            else:
                range_y = range_y1 if n == 0 else range_y2
        else:
            range_y = None

        if 'n_' in var_to_plot or 'p_' in var_to_plot:
            orig_y = var_to_plot.replace('n_', '').replace('p_', '').replace('_low', '').replace('_high', '').upper()
            fig.update_yaxes(title={'text': f'Normalised {orig_y}', 'font': {'size': font_size}}, secondary_y=n == 1, range=range_y)
            if not second_ticks:
                fig.update_yaxes(title={'text': '', 'font': {'size': font_size}}, secondary_y=True, range=range_y, showticklabels=False)
        elif var_to_plot == 'nnz':
            fig.update_yaxes(title={'text': 'Number of Edges in DAG', 'font': {'size': font_size}}, secondary_y=n == 1, range=range_y)
        elif range_y is not None:
            fig.update_yaxes(title={'text': var_to_plot.title(), 'font': {'size': font_size}}, secondary_y=n == 1, range=range_y)
        else:
            fig.update_yaxes(title={'text': var_to_plot.title(), 'font': {'size': font_size}}, secondary_y=n == 1)

    label_config = {
        ('precision', 'recall'): ('Precision', 'Recall', 0.65, 0.15),
        ('fdr', 'tpr'): ('FDR', 'TPR', 0.60, 0.12),
        ('p_shd', 'p_SID'): ('NSHD', 'NSID', 0.60, 0.12),
        ('p_shd', 'p_SID_low'): ('NSHD', 'NSID', 0.60, 0.12),
        ('p_shd', 'p_SID_high'): ('NSHD', 'NSID', 0.60, 0.12),
        ('p_SID_low', 'p_SID_high'): ('Best', 'Worst', 0.55, 0.18),
        ('p_shd', 'F1'): ('NSHD', 'F1', 0.63, 0.16),
    }
    key = tuple(vars_to_plot)
    name1, name2, total_width_frac, gap_frac = label_config.get(
        key,
        (vars_to_plot[0].upper(), vars_to_plot[1].upper() if len(vars_to_plot) > 1 else '', 0.60, 0.10)
    )

    label_names = [label for label in (name1, name2) if label]
    if not label_names:
        label_names = ['']
    label_count = len(label_names)

    unique_datasets = list(dict.fromkeys(all_sum['dataset'].tolist()))
    n_x_cat = max(len(unique_datasets), 1)
    cluster_width = 1.0 / n_x_cat
    total_tile_width = min(cluster_width * total_width_frac, cluster_width * 0.9)
    tile_width = total_tile_width / label_count
    if label_count == 1:
        gap = 0.0
    else:
        max_gap_available = max(cluster_width - total_tile_width, 0.0)
        gap = min(cluster_width * gap_frac, max_gap_available / (label_count - 1))
    cluster_padding = max((cluster_width - (tile_width * label_count + gap * (label_count - 1))) / 2, 0.0)

    top_y0 = 1.04
    top_y1 = 1.10
    text_y = (top_y0 + top_y1) / 2

    pad_lookup = {
        'Precision': 6, 'Recall': 8,
        'FDR': 5, 'TPR': 7,
        'NSHD': 9, 'NSID': 9,
        'Best': 9, 'Worst': 9,
        'F1': 11,
    }

    for dataset_index in range(n_x_cat):
        cluster_left = dataset_index * cluster_width + cluster_padding
        for label_index, label_text in enumerate(label_names):
            left = cluster_left + label_index * (tile_width + gap)
            right = left + tile_width
            
            # Make grey rectangles wider by expanding them outward
            rect_expansion = rect_exp
            rect_left = max(0, left - rect_expansion)
            rect_right = min(1, right + rect_expansion)
            
            pad = pad_lookup.get(label_text, 8)
            fig.add_shape(
                type='rect',
                xref='x domain',
                yref='y domain',
                x0=rect_left,
                x1=rect_right,
                y0=top_y0,
                y1=top_y1,
                line=dict(color='#E5ECF6', width=2),
                fillcolor='#E5ECF6',
                layer='below',
            )
            fig.add_annotation(
                xref='x domain',
                yref='y domain',
                x=(left + right) / 2,
                y=text_y,
                xanchor='center',
                yanchor='middle',
                text=f"{' ' * pad}{label_text}{' ' * pad}",
                showarrow=False,
                font=dict(size=font_size, color='black'),
            )

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html', '.jpeg'))

    fig.show()

###Plot Runtime
def plot_runtime(df, x_var_list, general_filter, names_dict, symbols_dict, colors_dict, methods,
                         share_y=False, save_figs=False, output_name="runtime.html", debug=False, font_size=20, plot_width=750, plot_height=300, model_aliases={}):
    cols = len(x_var_list)
    rows = 1
    fig = make_subplots(rows, cols, vertical_spacing=0.05, horizontal_spacing=0.01, shared_yaxes=True, shared_xaxes=True,)
    i = 1
    j = 1
    metric = 'elapsed_mean'
    if model_aliases == {}: model_aliases = names_dict

    # Apply optional row filter (e.g., "n_nodes <= 10"). If string is empty/None, keep df as-is.
    if isinstance(general_filter, str) and general_filter.strip():
        try:
            df = df.query(general_filter)
        except Exception:
            # Silently ignore invalid filters to preserve previous behaviour
            if debug:
                print(f"[plot_runtime] Ignoring invalid filter: {general_filter}")

    for x_var in x_var_list:
        group_list = [x_var]
        for model in methods:#df_to_plot['model_test'].unique():
            filter = f"model=='{names_dict[model]}'"
            tab = df.query(filter)
            tab_grouped = pd.DataFrame()
            for g in df[group_list].drop_duplicates().values:
                ## aggregate only if more than one obs in the group
                try:
                    relevant_row = list(tab.groupby(group_list)[metric].count().loc[g])[0]
                except:
                    relevant_row = 0
                if relevant_row > 1:
                    tab_grouped = pd.concat([tab_grouped, tab.groupby(group_list)[metric].agg(['mean','std']).loc[g].reset_index()], axis=0)
                else:
                    ##append the single obs
                    if relevant_row == 1:
                        single_tab = tab.query(f"{x_var}=={g[0]}")[[x_var,metric,metric.replace('mean','std')]]
                    else:
                        single_tab = pd.DataFrame([g[0], np.nan, np.nan]).T
                    single_tab.columns = [x_var,'mean','std']
                    single_tab[x_var] = single_tab[x_var].astype(int)
                    tab_grouped = pd.concat([tab_grouped, single_tab], axis=0)
            tab_grouped.reset_index(drop=True, inplace=True)
            tab_grouped.sort_values(by=[x_var], inplace=True)
            # Use categorical x by converting to string; this avoids mixed numeric/categorical axes.
            fig.add_trace(go.Scatter(x=tab_grouped[x_var].astype(str)
                                    ,y=tab_grouped[metric.split("_")[1]]
                                    ,error_y=dict(type='data', array=tab_grouped['std'])
                                    ,name=model_aliases[model], #legendgroup=f'group{i}{j}', 
                                    line=dict(color=colors_dict[model], width=2, simplify=True), mode='lines+markers', 
                                    marker=dict(size=8, symbol=symbols_dict[model], color=colors_dict[model]), 
                                    showlegend=(j==1 and i==1)), j, i)
        i += 1 if i < cols else 0

    fig.update_yaxes(matches='y', type="log")

    # # Force a single categorical x-axis with a stable category order derived from the data.
    # try:
    #     for c, x_var in enumerate(x_var_list):
    #         # Collect categories from the (possibly filtered) df
    #         cats = (
    #             pd.to_numeric(df[x_var], errors='coerce')
    #               .dropna()
    #               .astype(int)
    #               .sort_values()
    #               .unique()
    #         )
    #         cat_array = [str(v) for v in cats]
    #         this_xaxis = next(fig.select_xaxes(row=rows, col=c+1))
    #         this_xaxis.update(type='category', categoryorder='array', categoryarray=cat_array, side='bottom')

    #     # Hide any accidental secondary x-axes that Plotly may generate
    #     xaxes_all = list(fig.select_xaxes())
    #     for idx, ax in enumerate(xaxes_all):
    #         if idx > 0:
    #             ax.update(showticklabels=False, ticks='', showgrid=False, title='')
    # except Exception:
    #     # If anything goes wrong, leave Plotly's defaults
    #     if debug:
    #         print("[plot_runtime] Could not enforce category x-axis ordering.")

    for r in range(1,rows+1):
        this_yaxis = next(fig.select_yaxes(row = r, col = 1))
        this_yaxis.update(title='log(elapsed time [s])',title_standoff=0)
    for c,n in enumerate(["Number of Nodes (|V|)",
                        # "Proportional Sample Size (s=N/|V|)"
                        # "Sample Size (N)"
                        ]):
        this_xaxis = next(fig.select_xaxes(row = rows, col = c+1))
        this_xaxis.update(title=n,title_standoff=0)

    fig.update_layout(
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
        template='plotly_white',
        # autosize=True,
        width=plot_width, 
        height=plot_height,
        margin=dict(
            l=10,
            r=10,
            b=80,
            t=10,
        ),font=dict(size=font_size, family="Serif", color="black")
                    )

    for dl in range(0,len(fig.data)):
        fig.data[dl].error_y.thickness = 2
    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))
    fig.show()
