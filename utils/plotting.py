from plotly import graph_objects as go
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
    for method in ['Random', 'FGS', 'MCSL-MLP', 'NOTEARS-MLP', 'Max-PC', 'SPC (Ours)', 'ABAPC (Ours)']:
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