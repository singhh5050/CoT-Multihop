import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

# Set style for clean, presentation-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create a directory for improved visualizations
import os
if not os.path.exists('results/visualizations'):
    os.makedirs('results/visualizations')

# Define a consistent, colorblind-friendly color scheme
COLORS = {
    "Direct Answer": "#3366CC",  # Blue
    "Single-agent CoT": "#FF9900",  # Orange
    "Multi-agent CoT": "#109618"   # Green
}

# Load the data from summary tables
# Overall performance data
overall_data = pd.DataFrame({
    'Method': ['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT'],
    'Exact Match': [7.0, 17.0, 14.0],
    'F1 Score': [18.9, 30.1, 26.1],
    'CI_Low': [2.0, 9.6, 7.2],
    'CI_High': [12.0, 24.4, 20.8]
})

# Performance by hop count
hop2_data = pd.DataFrame({
    'Method': ['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT'],
    'Exact Match': [2.6, 23.1, 15.4],
    'F1 Score': [17.8, 40.5, 33.5],
    'CI_Low': [0.0, 9.9, 4.1],  # Clamped to non-negative
    'CI_High': [7.5, 36.3, 26.7],
    'Hop': [2, 2, 2]
})

hop3_data = pd.DataFrame({
    'Method': ['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT'],
    'Exact Match': [5.3, 10.5, 7.9],
    'F1 Score': [15.0, 20.9, 14.4],
    'CI_Low': [0.0, 0.8, 0.0],  # Clamped to non-negative
    'CI_High': [12.4, 20.3, 16.5],
    'Hop': [3, 3, 3]
})

hop4_data = pd.DataFrame({
    'Method': ['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT'],
    'Exact Match': [17.4, 17.4, 21.7],
    'F1 Score': [27.2, 27.6, 32.8],
    'CI_Low': [1.9, 1.9, 4.9],
    'CI_High': [32.9, 32.9, 38.6],
    'Hop': [4, 4, 4]
})

# Combine hop data
hop_data = pd.concat([hop2_data, hop3_data, hop4_data])

# Latency data
latency_data = pd.DataFrame({
    'Method': ['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT'] * 3,
    'Hop': [2, 2, 2, 3, 3, 3, 4, 4, 4],
    'Latency': [0.8, 2.9, 5.4, 0.6, 3.9, 7.2, 0.7, 4.1, 8.8]
})

# Dataset distribution data
distribution_data = pd.DataFrame({
    'Hop Count': [2, 3, 4],
    'Examples': [1252, 760, 405],
    'Percentage': [51.8, 31.4, 16.8]
})

# 1. IMPROVED DATASET COMPOSITION - Simple Bar Chart
def create_dataset_composition_chart():
    plt.figure(figsize=(10, 6))
    
    # Create bar chart for hop distribution
    ax = plt.bar(
        distribution_data['Hop Count'].astype(str) + '-hop', 
        distribution_data['Percentage'],
        color=sns.color_palette("Blues", 3),
        edgecolor='black',
        width=0.6
    )
    
    # Add counts and percentages as annotations
    for i, p in enumerate(ax):
        height = p.get_height()
        count = distribution_data['Examples'][i]
        plt.text(
            p.get_x() + p.get_width()/2.,
            height + 1,
            f'{height:.1f}%\n({count:,})',
            ha='center',
            fontweight='bold'
        )
    
    plt.title('MuSiQue Dataset Composition by Hop Count', fontsize=18)
    plt.ylabel('Percentage of Dataset (%)', fontsize=14)
    plt.ylim(0, 60)  # Enough room for annotations
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total sample size as footnote
    total = sum(distribution_data['Examples'])
    plt.figtext(0.5, 0.01, f'N = {total:,} total examples', 
                ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/dataset_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
# 2. IMPROVED PERFORMANCE BY HOP COUNT - Clear Line Chart
def create_performance_by_hop_chart():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Markers for each method to maintain consistency
    markers = {
        'Direct Answer': 'o', 
        'Single-agent CoT': 's', 
        'Multi-agent CoT': '^'
    }
    
    for ax_idx, metric in enumerate(['Exact Match', 'F1 Score']):
        ax = axes[ax_idx]
        
        for method in hop_data['Method'].unique():
            method_data = hop_data[hop_data['Method'] == method]
            
            # Plot the main line
            ax.plot(
                method_data['Hop'],
                method_data[metric],
                marker=markers[method],
                markersize=10,
                linewidth=2.5,
                color=COLORS[method],
                label=method
            )
        
        # Improve axis formatting
        ax.set_xlabel('Hop Count', fontsize=14)
        ax.set_ylabel(f'{metric} (%)', fontsize=14)
        ax.set_title(f'{metric} by Hop Count', fontsize=16)
        ax.set_xticks([2, 3, 4])
        ax.set_xlim(1.8, 4.2)
        ax.set_ylim(0, max(hop_data[metric].max() * 1.1, 50))  # Ensure reasonable y-limit
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if ax_idx == 0:
            # Place legend outside the first plot
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/performance_by_hop.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. IMPROVED OVERALL PERFORMANCE - Grouped Bar Chart (replacing radar)
def create_overall_performance_chart():
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(overall_data['Method']))
    width = 0.35
    
    # Create grouped bars for EM and F1
    plt.bar(
        x - width/2, 
        overall_data['Exact Match'], 
        width, 
        label='Exact Match',
        color=[COLORS[m] for m in overall_data['Method']], 
        alpha=0.8, 
        edgecolor='black'
    )
    
    plt.bar(
        x + width/2, 
        overall_data['F1 Score'], 
        width, 
        label='F1 Score',
        color=[COLORS[m] for m in overall_data['Method']], 
        alpha=0.4, 
        edgecolor='black', 
        hatch='//'
    )
    
    # Add data labels
    for i, method in enumerate(overall_data['Method']):
        plt.text(
            i - width/2, 
            overall_data['Exact Match'][i] + 0.5, 
            f"{overall_data['Exact Match'][i]}%", 
            ha='center', 
            va='bottom', 
            fontweight='bold',
            fontsize=11
        )
        
        plt.text(
            i + width/2, 
            overall_data['F1 Score'][i] + 0.5, 
            f"{overall_data['F1 Score'][i]}%", 
            ha='center', 
            va='bottom', 
            fontweight='bold',
            fontsize=11
        )
    
    plt.xlabel('')
    plt.ylabel('Score (%)', fontsize=14)
    plt.title('Overall Performance by Method', fontsize=18)
    plt.xticks(x, overall_data['Method'])
    plt.ylim(0, max(overall_data['F1 Score'].max(), overall_data['Exact Match'].max()) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. IMPROVED LATENCY CHART - Grouped by Hop
def create_latency_chart():
    # Create a simple bar chart for latency by method and hop count
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Reshape data for easier plotting
    pivot_latency = latency_data.pivot(index='Hop', columns='Method', values='Latency')
    
    # Set width and positions
    x = np.arange(len(pivot_latency.index))
    width = 0.25
    
    # Plot bars in a consistent order
    for i, method in enumerate(['Direct Answer', 'Single-agent CoT', 'Multi-agent CoT']):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, 
            pivot_latency[method], 
            width, 
            label=method,
            color=COLORS[method],
            edgecolor='black',
            alpha=0.8
        )
        
        # Add latency values as labels
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f'{height}s',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=10
            )
    
    # Style the main plot
    ax.set_ylabel('Latency (seconds)', fontsize=14)
    ax.set_title('Inference Latency by Method and Hop Count', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{hop}-hop' for hop in pivot_latency.index])
    ax.set_ylim(0, max(pivot_latency.values.flatten()) * 1.2)  # Add space for labels
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Move legend outside to the right
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/latency_by_hop.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. IMPROVED TRADEOFF CHART - Vertical Arrangement of Scatter Plots
def create_performance_latency_tradeoff():
    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
    hop_titles = {2: '2-hop', 3: '3-hop', 4: '4-hop'}
    
    # Compile tradeoff data
    tradeoff_data = []
    for method in hop_data['Method'].unique():
        for hop in hop_data['Hop'].unique():
            perf_row = hop_data[(hop_data['Method'] == method) & (hop_data['Hop'] == hop)]
            latency_row = latency_data[(latency_data['Method'] == method) & (latency_data['Hop'] == hop)]
            
            if not perf_row.empty and not latency_row.empty:
                tradeoff_data.append({
                    'Method': method,
                    'Hop': hop,
                    'F1 Score': perf_row['F1 Score'].values[0],
                    'Latency': latency_row['Latency'].values[0],
                    'Efficiency': perf_row['F1 Score'].values[0] / latency_row['Latency'].values[0]
                })
    
    tradeoff_df = pd.DataFrame(tradeoff_data)
    
    # Plot each hop in its own panel
    for i, hop in enumerate([2, 3, 4]):
        ax = axes[i]
        subset = tradeoff_df[tradeoff_df['Hop'] == hop]
        
        # Plot scatter points
        for j, method in enumerate(subset['Method']):
            row = subset[subset['Method'] == method]
            ax.scatter(
                row['Latency'], 
                row['F1 Score'],
                s=200,  # Increased marker size
                color=COLORS[method],
                edgecolors='black',
                linewidth=1.5,
                alpha=0.8,
                label=method
            )
            
            # Add method labels
            ax.annotate(
                method.split('-')[0],  # Short version of the method name
                (row['Latency'].values[0], row['F1 Score'].values[0]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontweight='bold',
                fontsize=12
            )
        
        # Format each subplot
        ax.set_title(f'{hop_titles[hop]} Questions', fontsize=16)
        if i == 2:  # Only add x-label to the bottom plot
            ax.set_xlabel('Latency (seconds)', fontsize=14)
        ax.set_ylabel('F1 Score (%)', fontsize=14)
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 45)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add efficiency annotation only to the best performer
        best_row = subset.loc[subset['F1 Score'].idxmax()]
        ax.annotate(
            f"Efficiency: {best_row['Efficiency']:.1f} F1/s",
            (best_row['Latency'], best_row['F1 Score']),
            xytext=(10, -20),
            textcoords='offset points',
            fontsize=12,
            color='darkred',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
        )
        
        # Add legend to each subplot
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/performance_latency_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualization functions
create_dataset_composition_chart()
create_performance_by_hop_chart()
create_overall_performance_chart()
create_latency_chart()
create_performance_latency_tradeoff()

print("Visualizations created successfully in the 'results/visualizations' directory.") 