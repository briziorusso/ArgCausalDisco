import matplotlib.pyplot as plt
import numpy as np

def create_thesis_plots(df, save_path=None, figsize=(14, 6), dpi=None):
    """
    Create publication-ready bar charts for Mean Average Precision and Mean NDCG.
    
    Parameters:
    df: pandas DataFrame with columns: dataset, S_weight, cit_method, mean_average_precision, mean_ndcg
    save_path: str, path to save figures (optional, will save as SVG if provided)
    figsize: tuple, figure size in inches
    dpi: int, resolution for saved figures
    """
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    datasets = sorted(df['dataset'].unique())
    cit_methods = sorted(df['cit_method'].unique())
    # Sort weights True then False to control bar order
    weight_options = sorted(df['S_weight'].unique(), reverse=True)
    
    # Create method combinations dynamically
    methods = [(method, weight) for method in cit_methods for weight in weight_options]
    
    # Generate colors automatically based on unique methods
    base_colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#E67E22']
    unique_methods = sorted(df['cit_method'].unique())
    method_colors = {method: base_colors[i % len(base_colors)] for i, method in enumerate(unique_methods)}

    colors = {}
    patterns = {}
    
    for method, weight in methods:
        colors[(method, weight)] = method_colors[method]
        if weight:  # Penalised
            patterns[(method, weight)] = '////'
        else:  # Not Penalised
            patterns[(method, weight)] = None
            
    # Generate method labels dynamically
    method_labels = [f'{method.upper()}{"(P)" if weight else ""}' 
                     for method, weight in methods]
    
    # Set up bar positions
    x = np.arange(len(datasets))
    width = 0.2
    
    # Create both plots
    metrics = [
        ('mean_average_precision', 'Mean Average Precision', 'AP'),
        ('mean_ndcg', 'Mean NDCG', 'NDCG')
    ]
    
    for metric_col, metric_title, metric_short in metrics:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot bars for each method
        for i, ((method, weighted), label) in enumerate(zip(methods, method_labels)):
            values = []
            for dataset in datasets:
                subset = df[(df['dataset'] == dataset) & 
                            (df['cit_method'] == method) & 
                            (df['S_weight'] == weighted)]
                if len(subset) > 0:
                    values.append(subset[metric_col].iloc[0])
                else:
                    values.append(0)  # Handle missing data
            
            bars = ax.bar(x + i * width, values, width, 
                          label=label, 
                          color=colors[(method, weighted)],
                          hatch=patterns[(method, weighted)],
                          edgecolor='black',
                          linewidth=0.5,
                          alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{value:.3f}',
                            ha='center', va='bottom', fontsize=9, rotation=90)
        
        # Customize the plot
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_title(f'{metric_title} by Dataset and Method', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([d.capitalize() for d in datasets])
        
        # Set y-axis limits with some padding
        if metric_col == 'mean_ndcg':
            ax.set_ylim(0.91, 1.01)
        else:
            y_min = df[metric_col].min() - 0.02
            y_max = df[metric_col].max() + 0.05
            ax.set_ylim(y_min, y_max)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            filename = f"{save_path}_{metric_short.lower()}_comparison.svg"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"Saved: {filename}")
        
        plt.show()

def create_side_by_side_plot(df, save_path=None, figsize=(16, 6), dpi=None):
    """
    Create a side-by-side comparison plot with both metrics in one figure.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    datasets = sorted(df['dataset'].unique())
    cit_methods = sorted(df['cit_method'].unique())
    # Sort weights True then False to control bar order
    weight_options = sorted(df['S_weight'].unique(), reverse=True)
    
    methods = [(method, weight) for method in cit_methods for weight in weight_options]
    
    # Generate colors and patterns automatically
    base_colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#E67E22']
    unique_methods = sorted(df['cit_method'].unique())
    method_colors = {method: base_colors[i % len(base_colors)] for i, method in enumerate(unique_methods)}

    colors = {}
    patterns = {}
    
    for method, weight in methods:
        colors[(method, weight)] = method_colors[method]
        if weight:  # Penalised
            patterns[(method, weight)] = '////'
        else:  # Not Penalised
            patterns[(method, weight)] = None
            
    # Generate method labels
    method_labels = [f'{method.upper()}{"(P)" if weight else ""}' 
                     for method, weight in methods]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    metrics = [
        ('mean_average_precision', 'Mean Average Precision', ax1),
        ('mean_ndcg', 'Mean NDCG', ax2)
    ]
    
    for metric_col, metric_title, ax in metrics:
        for i, ((method, weighted), label) in enumerate(zip(methods, method_labels)):
            values = []
            for dataset in datasets:
                subset = df[(df['dataset'] == dataset) & 
                            (df['cit_method'] == method) & 
                            (df['S_weight'] == weighted)]
                if len(subset) > 0:
                    values.append(subset[metric_col].iloc[0])
                else:
                    values.append(0)
            
            bars = ax.bar(x + i * width, values, width, 
                          label=label if ax == ax1 else "",
                          color=colors[(method, weighted)],
                          hatch=patterns[(method, weighted)],
                          edgecolor='black',
                          linewidth=0.5,
                          alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                            f'{value:.4f}',
                            ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_title(metric_title, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([d.capitalize() for d in datasets], rotation=45, ha='right')
        
        # Set y-axis limits
        if metric_col == 'mean_ndcg':
            ax.set_ylim(0.91, 1.01)
        else:
            y_min = df[metric_col].min() - 0.02
            y_max = df[metric_col].max() + 0.05
            ax.set_ylim(y_min, y_max)
            
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Add legend to the figure
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        filename = f"{save_path}_combined_comparison.svg"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
    
    plt.show()
