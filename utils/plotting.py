from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
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

def bar_chart_plotly(all_sum, var_to_plot, names_dict, colors_dict, save_figs=False, output_name="bar_chart.html", debug=False):
    fig = go.Figure()
    # for dataset_name in ['asia','cancer','earthquake','sachs','survey','alarm','child','insurance','hepar2']:
    # for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
    for method in ['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)']:
        trace_name = 'True Graph Size' if var_to_plot=='nnz' and method=='Random' else method
        fig.add_trace(go.Bar(x=all_sum[(all_sum.model==method)]['dataset'], 
                                y=all_sum[(all_sum.model==method)][var_to_plot+'_mean'], 
                                error_y=dict(type='data', array=all_sum[(all_sum.model==method)][var_to_plot+'_std'], visible=True),
                                name=trace_name,
                                marker_color=colors_dict[list(names_dict.keys())[list(names_dict.values()).index(method)]],
                                opacity=0.6,
                            #  width=0.1
                                )
                                )
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

    fig.show()

def double_bar_chart_plotly(all_sum, vars_to_plot, names_dict, colors_dict, 
                            methods=['Random', 'FGS', 'NOTEARS-MLP', 'Shapley-PC', 'ABAPC (Ours)'],
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
                r=40,
                b=80,
                t=20,
                # pad=10
            ),hovermode='x unified',
            font=dict(size=20, family="Serif", color="black"),
            yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=True, zeroline=True),
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
    


    for n, var_to_plot in enumerate(vars_to_plot):
        if vars_to_plot == ['precision', 'recall']:
            range_y = [0, 1.3]
        elif vars_to_plot == ['fdr', 'tpr']:
            range_y = [0, 1]
        elif vars_to_plot == ['p_shd', 'p_SID']:
            range_y = [0, 1.9] if n==0 else [0, max(all_sum['p_SID_mean'])+.5]
        if 'n_' in var_to_plot or 'p_' in var_to_plot:
            orig_y = var_to_plot.replace('n_','').replace('p_','').upper()
            fig.update_yaxes(title={'text':f'Normalised {orig_y} = {orig_y} / Number of Edges in DAG','font':{'size':20}}, secondary_y=n==1, range=range_y)
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
    elif vars_to_plot == ['p_shd', 'p_SID']:
        name1 = 'NSHD'
        name2 = 'NSID'
        lin_space=9
        nl_space=9

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

    fig.show()