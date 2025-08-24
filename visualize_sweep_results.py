#!/usr/bin/env python3
"""
Parameter Sweep Results Visualization Script

Generates comprehensive graphs and analysis based on the parameter sweep results.
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.usetex'] = False  # Use matplotlib's mathtext instead of LaTeX
plt.rcParams['font.family'] = 'DejaVu Sans'

# Disable interactive mode - only save figures
plt.ioff()

# Paths
RESULTS_DIR = Path("parameter_sweep_results")
OUTPUT_DIR = Path("parameter_sweep_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all the sweep results data"""
    print("Loading parameter sweep data...")
    
    # Load summary statistics
    with open(RESULTS_DIR / "summary_statistics.json", 'r') as f:
        summary_data = json.load(f)
    
    # Load progress summary
    with open(RESULTS_DIR / "progress_summary.json", 'r') as f:
        progress_data = json.load(f)
    
    # Load detailed results if available
    detailed_data = None
    try:
        with open(RESULTS_DIR / "detailed_results.pkl", 'rb') as f:
            detailed_data = pickle.load(f)
    except:
        print("Warning: Could not load detailed results pickle file")
    
    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    print(f"Loaded {len(df)} parameter configurations")
    print(f"Total experiments: {progress_data['experiments_completed']}")
    
    return df, progress_data, detailed_data

def create_parameter_performance_heatmaps(df):
    """Create heatmaps showing performance across parameter combinations"""
    print("Creating parameter performance heatmaps...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Parameter Performance Heatmaps - Mean Final Accuracy', fontsize=16)
    
    # Get unique parameter values
    tau_vals = sorted(df['tau'].unique())
    lr_vals = sorted(df['lr'].unique())
    eta_vals = sorted(df['eta'].unique())
    
    # 1. Tau vs LR (averaged over beta and eta)
    pivot_tau_lr = df.groupby(['tau', 'lr'])['mean_final_accuracy'].mean().unstack()
    sns.heatmap(pivot_tau_lr, annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title(r'$\tau$ vs Learning Rate')
    axes[0].set_ylabel(r'$\tau$')
    axes[0].set_xlabel(r'Learning Rate $\alpha$')
    
    # 2. Tau vs Eta (averaged over beta and lr)
    pivot_tau_eta = df.groupby(['tau', 'eta'])['mean_final_accuracy'].mean().unstack()
    sns.heatmap(pivot_tau_eta, annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title(r'$\tau$ vs $\eta$')
    axes[1].set_ylabel(r'$\tau$')
    axes[1].set_xlabel(r'$\eta$ (Nudging Strength)')
    
    # 3. LR vs Eta (averaged over tau and beta)
    pivot_lr_eta = df.groupby(['lr', 'eta'])['mean_final_accuracy'].mean().unstack()
    sns.heatmap(pivot_lr_eta, annot=True, fmt='.3f', cmap='RdYlBu_r', vmin=0, vmax=1, ax=axes[2])
    axes[2].set_title(r'Learning Rate vs $\eta$')
    axes[2].set_ylabel(r'Learning Rate $\alpha$')
    axes[2].set_xlabel(r'$\eta$ (Nudging Strength)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_distributions(df):
    """Create box plots for parameter distributions"""
    print("Creating parameter distribution plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Parameter Performance Distributions', fontsize=16)
    
    # Box plots only (excluding beta)
    sns.boxplot(data=df, x='tau', y='mean_final_accuracy', ax=axes[0])
    axes[0].set_title(r'Accuracy by $\tau$')
    axes[0].set_xlabel(r'$\tau$')
    axes[0].set_ylabel('Mean Final Accuracy')
    
    sns.boxplot(data=df, x='lr', y='mean_final_accuracy', ax=axes[1])
    axes[1].set_title(r'Accuracy by Learning Rate')
    axes[1].set_xlabel(r'Learning Rate $\alpha$')
    axes[1].set_ylabel('Mean Final Accuracy')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='eta', y='mean_final_accuracy', ax=axes[2])
    axes[2].set_title(r'Accuracy by $\eta$')
    axes[2].set_xlabel(r'$\eta$ (Nudging Strength)')
    axes[2].set_ylabel('Mean Final Accuracy')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_analysis(df):
    """Analyze convergence patterns with line plots"""
    print("Creating convergence analysis plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Convergence Analysis', fontsize=16)
    
    # 1. Convergence rate by tau (averaged over other parameters)
    tau_convergence = df.groupby('tau').agg({
        'convergence_rate': ['mean', 'std'],
        'mean_final_accuracy': ['mean', 'std']
    }).round(4)
    
    tau_values = tau_convergence.index
    conv_means = tau_convergence['convergence_rate']['mean']
    conv_stds = tau_convergence['convergence_rate']['std']
    acc_means = tau_convergence['mean_final_accuracy']['mean'] 
    acc_stds = tau_convergence['mean_final_accuracy']['std']
    
    # Plot convergence rate vs tau
    axes[0].errorbar(tau_values, conv_means, yerr=conv_stds, 
                     marker='o', linewidth=2, capsize=5, label='Convergence Rate')
    axes[0].set_xlabel(r'$\tau$')
    axes[0].set_ylabel('Convergence Rate')
    axes[0].set_title(r'Convergence Rate vs $\tau$')
    axes[0].grid(True, alpha=0.3)
    
    # Create second y-axis for accuracy
    ax0_twin = axes[0].twinx()
    ax0_twin.errorbar(tau_values, acc_means, yerr=acc_stds, 
                      marker='s', linewidth=2, capsize=5, color='red', label='Final Accuracy')
    ax0_twin.set_ylabel('Mean Final Accuracy', color='red')
    ax0_twin.tick_params(axis='y', labelcolor='red')
    
    # 2. Convergence rate by learning rate (averaged over other parameters)  
    lr_convergence = df.groupby('lr').agg({
        'convergence_rate': ['mean', 'std'],
        'mean_final_accuracy': ['mean', 'std']
    }).round(4)
    
    lr_values = lr_convergence.index
    conv_means_lr = lr_convergence['convergence_rate']['mean']
    conv_stds_lr = lr_convergence['convergence_rate']['std']
    acc_means_lr = lr_convergence['mean_final_accuracy']['mean']
    acc_stds_lr = lr_convergence['mean_final_accuracy']['std']
    
    # Plot convergence rate vs learning rate (log scale)
    axes[1].errorbar(lr_values, conv_means_lr, yerr=conv_stds_lr, 
                     marker='o', linewidth=2, capsize=5, label='Convergence Rate')
    axes[1].set_xlabel(r'Learning Rate $\alpha$')
    axes[1].set_ylabel('Convergence Rate')
    axes[1].set_title(r'Convergence Rate vs Learning Rate $\alpha$')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Create second y-axis for accuracy
    ax1_twin = axes[1].twinx()
    ax1_twin.errorbar(lr_values, acc_means_lr, yerr=acc_stds_lr, 
                      marker='s', linewidth=2, capsize=5, color='red', label='Final Accuracy')
    ax1_twin.set_ylabel('Mean Final Accuracy', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_best_worst_comparison(df):
    """Compare best and worst performing configurations"""
    print("Creating best vs worst performance comparison...")
    
    # Get top 10 and bottom 10 configurations
    top_configs = df.nlargest(10, 'mean_final_accuracy')
    bottom_configs = df.nsmallest(10, 'mean_final_accuracy')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Best vs Worst Performing Configurations', fontsize=16)
    
    # 1. Top 10 configurations
    top_labels = [f"τ={row['tau']}, β={row['beta']}, α={row['lr']:.0e}, η={row['eta']}" 
                  for _, row in top_configs.iterrows()]
    
    bars1 = axes[0,0].barh(range(len(top_configs)), top_configs['mean_final_accuracy'])
    axes[0,0].set_yticks(range(len(top_configs)))
    axes[0,0].set_yticklabels(top_labels, fontsize=10)
    axes[0,0].set_xlabel('Mean Final Accuracy')
    axes[0,0].set_title('Top 10 Configurations')
    axes[0,0].invert_yaxis()
    
    # Add error bars
    axes[0,0].errorbar(top_configs['mean_final_accuracy'], range(len(top_configs)),
                      xerr=top_configs['std_final_accuracy'], fmt='none', color='black', alpha=0.7)
    
    # 2. Bottom 10 configurations
    bottom_labels = [f"τ={row['tau']}, β={row['beta']}, α={row['lr']:.0e}, η={row['eta']}" 
                     for _, row in bottom_configs.iterrows()]
    
    bars2 = axes[0,1].barh(range(len(bottom_configs)), bottom_configs['mean_final_accuracy'])
    axes[0,1].set_yticks(range(len(bottom_configs)))
    axes[0,1].set_yticklabels(bottom_labels, fontsize=10)
    axes[0,1].set_xlabel('Mean Final Accuracy')
    axes[0,1].set_title('Bottom 10 Configurations')
    axes[0,1].invert_yaxis()
    
    # Add error bars
    axes[0,1].errorbar(bottom_configs['mean_final_accuracy'], range(len(bottom_configs)),
                      xerr=bottom_configs['std_final_accuracy'], fmt='none', color='black', alpha=0.7)
    
    # 3. Parameter frequency in top configurations
    top_param_counts = {
        'τ': top_configs['tau'].value_counts(),
        'β': top_configs['beta'].value_counts(),
        'α': top_configs['lr'].value_counts(),
        'η': top_configs['eta'].value_counts()
    }
    
    param_names = list(top_param_counts.keys())
    x_pos = range(len(param_names))
    
    # Count most frequent values
    most_frequent = [counts.iloc[0] if len(counts) > 0 else 0 for counts in top_param_counts.values()]
    
    axes[1,0].bar(param_names, most_frequent)
    axes[1,0].set_title('Parameter Frequency in Top 10')
    axes[1,0].set_xlabel('Parameter')
    axes[1,0].set_ylabel('Count in Top 10')
    
    # 4. Accuracy vs standard deviation
    axes[1,1].scatter(df['mean_final_accuracy'], df['std_final_accuracy'], 
                     alpha=0.6, s=50)
    axes[1,1].set_xlabel('Mean Final Accuracy')
    axes[1,1].set_ylabel('Standard Deviation')
    axes[1,1].set_title('Accuracy vs Variability')
    
    # Highlight best and worst
    axes[1,1].scatter(top_configs['mean_final_accuracy'], top_configs['std_final_accuracy'], 
                     color='green', s=100, alpha=0.8, label='Top 10')
    axes[1,1].scatter(bottom_configs['mean_final_accuracy'], bottom_configs['std_final_accuracy'], 
                     color='red', s=100, alpha=0.8, label='Bottom 10')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_worst_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_interaction_plots(df):
    """Create interaction plots between parameters with error bars (excluding beta)"""
    print("Creating parameter interaction plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Parameter Interactions', fontsize=16)
    
    # 1. Learning Rate and Eta interaction
    for lr in sorted(df['lr'].unique()):
        lr_data = df[df['lr'] == lr]
        lr_stats = lr_data.groupby('eta')['mean_final_accuracy'].agg(['mean', 'std']).reset_index()
        lr_stats['se'] = lr_stats['std'] / np.sqrt(lr_data.groupby('eta').size().values)
        
        axes[0].errorbar(lr_stats['eta'], lr_stats['mean'], yerr=lr_stats['se'], 
                        marker='o', label=f'α={lr:.0e}', capsize=5)
    
    axes[0].set_xlabel(r'$\eta$ (Nudging Strength)')
    axes[0].set_ylabel('Mean Final Accuracy')
    axes[0].set_title(r'Learning Rate $\alpha$-$\eta$ Interaction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Tau and Learning Rate interaction
    for tau in sorted(df['tau'].unique()):
        tau_data = df[df['tau'] == tau]
        tau_stats = tau_data.groupby('lr')['mean_final_accuracy'].agg(['mean', 'std']).reset_index()
        tau_stats['se'] = tau_stats['std'] / np.sqrt(tau_data.groupby('lr').size().values)
        
        axes[1].errorbar(tau_stats['lr'], tau_stats['mean'], yerr=tau_stats['se'], 
                        marker='o', label=f'τ={tau}', capsize=5)
    
    axes[1].set_xlabel(r'Learning Rate $\alpha$ (log scale)')
    axes[1].set_ylabel('Mean Final Accuracy')
    axes[1].set_title(r'$\tau$-Learning Rate $\alpha$ Interaction')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_interactions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics_table(df, progress_data):
    """Create a comprehensive summary table"""
    print("Creating summary statistics table...")
    
    # Overall statistics
    best_config = df.loc[df['mean_final_accuracy'].idxmax()]
    worst_config = df.loc[df['mean_final_accuracy'].idxmin()]
    
    # Parameter-wise statistics
    param_stats = {}
    for param in ['tau', 'beta', 'lr', 'eta']:
        param_means = df.groupby(param)['mean_final_accuracy'].mean()
        best_param_val = param_means.idxmax()
        worst_param_val = param_means.idxmin()
        param_stats[param] = {
            'best_value': best_param_val,
            'best_accuracy': param_means[best_param_val],
            'worst_value': worst_param_val,
            'worst_accuracy': param_means[worst_param_val]
        }
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Parameter Sweep Summary Statistics', fontsize=16)
    
    # 1. Overall performance distribution
    axes[0,0].hist(df['mean_final_accuracy'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(df['mean_final_accuracy'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df["mean_final_accuracy"].mean():.4f}')
    axes[0,0].axvline(best_config['mean_final_accuracy'], color='green', linestyle='--',
                     label=f'Best: {best_config["mean_final_accuracy"]:.4f}')
    axes[0,0].set_xlabel('Mean Final Accuracy')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Overall Accuracy Distribution')
    axes[0,0].legend()
    
    # 2. Parameter importance (range of performance)
    param_ranges = []
    param_names = []
    param_symbols = [r'$\tau$', r'$\beta$', r'$\alpha$', r'$\eta$']
    for param in ['tau', 'beta', 'lr', 'eta']:
        param_means = df.groupby(param)['mean_final_accuracy'].mean()
        param_ranges.append(param_means.max() - param_means.min())
        param_names.append(param)
    
    bars = axes[0,1].bar(param_symbols, param_ranges)
    axes[0,1].set_xlabel('Parameter')
    axes[0,1].set_ylabel('Accuracy Range')
    axes[0,1].set_title('Parameter Importance (Performance Range)')
    
    # Add value labels on bars
    for bar, value in zip(bars, param_ranges):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{value:.4f}', ha='center', va='bottom')
    
    # 3. Convergence summary
    convergence_counts = {
        'Converged': (df['convergence_rate'] > 0).sum(),
        'Did not converge': (df['convergence_rate'] == 0).sum()
    }
    
    axes[1,0].pie(convergence_counts.values(), labels=convergence_counts.keys(), 
                 autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Convergence Success Rate')
    
    # 4. Text summary
    axes[1,1].axis('off')
    summary_text = f"""
PARAMETER SWEEP SUMMARY
======================
Total Configurations: {len(df)}
Total Experiments: {progress_data['experiments_completed']}
Runs per Config: {df['n_runs'].iloc[0]}

BEST CONFIGURATION:
τ={best_config['tau']}, β={best_config['beta']}
lr={best_config['lr']:.0e}, η={best_config['eta']}
Accuracy: {best_config['mean_final_accuracy']:.4f} ± {best_config['std_final_accuracy']:.4f}

WORST CONFIGURATION:
τ={worst_config['tau']}, β={worst_config['beta']}
lr={worst_config['lr']:.0e}, η={worst_config['eta']}
Accuracy: {worst_config['mean_final_accuracy']:.4f} ± {worst_config['std_final_accuracy']:.4f}

PARAMETER ANALYSIS:
Best τ: {param_stats['tau']['best_value']} (acc: {param_stats['tau']['best_accuracy']:.4f})
Best β: {param_stats['beta']['best_value']} (acc: {param_stats['beta']['best_accuracy']:.4f})
Best lr: {param_stats['lr']['best_value']:.0e} (acc: {param_stats['lr']['best_accuracy']:.4f})
Best η: {param_stats['eta']['best_value']} (acc: {param_stats['eta']['best_accuracy']:.4f})

Overall Mean Accuracy: {df['mean_final_accuracy'].mean():.4f}
Overall Std Accuracy: {df['mean_final_accuracy'].std():.4f}
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_config, worst_config, param_stats

def main():
    """Main function to generate all visualizations"""
    print("=" * 60)
    print("PARAMETER SWEEP RESULTS VISUALIZATION")
    print("=" * 60)
    
    # Load data
    df, progress_data, detailed_data = load_data()
    
    # Generate selected visualizations
    create_parameter_performance_heatmaps(df)
    create_parameter_distributions(df)
    create_convergence_analysis(df)
    create_parameter_interaction_plots(df)
    
    # Get best configuration for summary
    best_config = df.loc[df['mean_final_accuracy'].idxmax()]
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    
    print(f"\nBEST CONFIGURATION:")
    print(f"  tau={best_config['tau']}, beta={best_config['beta']}, lr={best_config['lr']:.0e}, eta={best_config['eta']}")
    print(f"  Accuracy: {best_config['mean_final_accuracy']:.6f} ± {best_config['std_final_accuracy']:.6f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
