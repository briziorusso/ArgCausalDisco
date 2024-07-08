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

def bar_chart_plotly(all_sum, var_to_plot, names_dict, colors_dict, methods, save_figs=False, output_name="bar_chart.html", debug=False):
    fig = go.Figure()
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    # for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
    for method in methods:#['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)']:
        trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method
        if 'log' in var_to_plot:
            var_to_plot = var_to_plot.replace('log_','')
            var_to_plot = "log(Elapsed Time)" if "lapsed" in var_to_plot else "log("+var_to_plot+")"
            trace_name = 'Log '+trace_name
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                opacity=0.6,
                            #  width=0.1
                                ))
            fig.update_yaxes(type="log")
        else:
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                    y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                    error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                    name=trace_name,
                                    marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                    opacity=0.6,
                                #  width=0.1
                                    ))
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.1, # gap between bars of the same location coordinate.)
            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1),
            template='plotly_white',
            # autosize=True,
            width=1600, 
            height=700,
            margin=dict(
                l=40,
                r=40,
                b=80,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=20, family="Serif", color="black")
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0,
        yanchor="bottom",
        y=-0.05,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=20,
                    color="Black"
                    )
        )

    if 'n_' in var_to_plot or 'p_' in var_to_plot:
        orig_y = var_to_plot.replace('n_','').replace('p_','').upper()
        fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':20}})
    elif var_to_plot=='nnz':
        orig_y = 'Number of Edges in DAG'
        fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':20}})
    else:
        fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':20}})

    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))
    fig.show()

def double_bar_chart_plotly(all_sum, vars_to_plot, names_dict, colors_dict, 
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)'],
                            range_y1=None, range_y2=None,
                            save_figs=False, output_name="bar_chart.html", debug=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    # for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
    for n, var_to_plot in enumerate(vars_to_plot):
        for m, method in enumerate(methods):
            trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method#+' '+var_to_plot
            fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                yaxis=f"y{n+1}",
                                offsetgroup=m+len(methods)*n+(1*n),
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
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
    second_ticks = False if all('SID' in var for var in vars_to_plot) else True
    # Change the bar mode
    fig.update_layout(barmode='group',
                        bargap=0.15, # gap between bars of adjacent location coordinates.
                        bargroupgap=0.1, # gap between bars of the same location coordinate.)

            legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.1),
            template='plotly_white',
            # autosize=True,
            width=1600, 
            height=700,
            margin=dict(
                l=40,
                r=00,
                b=70,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=20, family="Serif", color="black"),
            yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=second_ticks, zeroline=True),
            )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        xanchor="center",
        x=0,
        yanchor="bottom",
        y=-0.08,
        text=f"Dataset:",
        showarrow=False,    
        font=dict(
                    family="Serif",
                    size=20,
                    color="Black"
                    )
        )
    
    for n, var_to_plot in enumerate(vars_to_plot):
        if vars_to_plot == ['precision', 'recall']:
            if range_y1 is None:
                range_y = [0, 1.3]
            else:
                range_y = range_y1
        elif vars_to_plot == ['fdr', 'tpr']:
            range_y = [0, 1]
        elif 'shd' in var_to_plot or 'SID' in var_to_plot:
            if range_y1 is None and range_y2 is None:
                if 'high' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_high_mean'])+.3]
                elif 'low' in var_to_plot:
                    range_y = [0, max(all_sum['p_SID_low_mean'])+.3]
                else:
                    range_y = [0, 2] if n==0 else [0, max(all_sum['p_SID_mean'])+.3]
            else:
                range_y = range_y1 if n==0 else range_y2
        if 'n_' in var_to_plot or 'p_' in var_to_plot:
            orig_y = var_to_plot.replace('n_','').replace('p_','').replace('_low','').replace('_high','').upper()
            fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':20}}, secondary_y=n==1, range=range_y)
            if second_ticks == False:
                fig.update_yaxes(title={'text':'','font':{'size':20}}, secondary_y=True, range=range_y, showticklabels=False)
        elif var_to_plot=='nnz':
            orig_y = 'Number of Edges in DAG'
            fig.update_yaxes(title={'text':f'{orig_y}','font':{'size':20}}, secondary_y=n==1, range=range_y)
        else:
            fig.update_yaxes(title={'text':f'{var_to_plot.title()}','font':{'size':20}}, secondary_y=n==1, range=range_y)

    start_pos = 0.017
    intra_dis = 0.12
    inter_dis = 0.13

    if vars_to_plot == ['precision', 'recall']:
        name1 = 'Precision'
        name2 = 'Recall'
        lin_space=6
        nl_space=8
        intra_dis = 0.115
        inter_dis = 0.135
    elif vars_to_plot == ['fdr', 'tpr']:
        name1 = 'FDR'
        name2 = 'TPR'
        lin_space=5
        nl_space=7
    elif vars_to_plot == ['p_shd', 'p_SID'] or vars_to_plot == ['p_shd', 'p_SID_low'] or vars_to_plot == ['p_shd', 'p_SID_high']:
        name1 = 'NSHD'
        name2 = 'NSID'
        lin_space=9
        nl_space=9
    elif vars_to_plot == ['p_SID_low', 'p_SID_high']:
        name1 = 'Low'
        name2 = 'High'
        lin_space=11
        nl_space=11
        intra_dis = 0.115
        inter_dis = 0.135
    elif vars_to_plot == ['p_shd', 'F1']:
        name1 = 'NSHD'
        name2 = 'F1'
        lin_space=9
        nl_space=11

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
            y=1.015,
                    text=f"{' '*lin_space}{name1}{' '*(lin_space)}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=20,
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
            y=1.015,
                    text=f"{' '*(nl_space)}{name2}{' '*nl_space}",
            showarrow=False,    
            font=dict(
                # family="Courier New, monospace",
                size=20,
                color="black"
                )
        , bordercolor='#E5ECF6'
        , borderwidth=2
        , bgcolor="#E5ECF6"
        , opacity=0.8
                )


    if save_figs:
        fig.write_html(output_name)
        fig.write_image(output_name.replace('.html','.jpeg'))

    fig.show()

###Plot Runtime
def plot_runtime(df, x_var_list, general_filter, names_dict, symbols_dict, colors_dict, methods,
                         share_y=False, save_figs=False, output_name="runtime.html", debug=False, font_size=20, plot_width=750, plot_height=300):
    cols = len(x_var_list)
    rows = 1
    fig = make_subplots(rows, cols, vertical_spacing=0.05, horizontal_spacing=0.01, shared_yaxes=True, shared_xaxes=True,)
    i = 1
    j = 1
    metric = 'elapsed_mean'

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
            fig.add_trace(go.Scatter(x=tab_grouped[x_var].astype(str)
                                    ,y=tab_grouped[metric.split("_")[1]]
                                    ,error_y=dict(type='data', array=tab_grouped['std'])
                                    ,name=names_dict[model], #legendgroup=f'group{i}{j}', 
                                    line=dict(color=colors_dict[model], width=2, simplify=True), mode='lines+markers', 
                                    marker=dict(size=8, symbol=symbols_dict[model], color=colors_dict[model]), 
                                    showlegend=(j==1 and i==1)), j, i)
        i += 1 if i < cols else 0

    fig.update_yaxes(matches='y', type="log")

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