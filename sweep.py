import random
import multiprocessing as mp
import itertools
import pickle
import json
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add the clln_mimicks directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import *
from beta_arch import ParallelBetaIsingEdgeCLLN

# Experimental parameters
PARAM_GRID = {
    'tau': [5, 10, 20, 50, 100],
    'beta': [1.0], 
    'lr': [1e-5, 1e-4, 1e-3], 
    'eta': [0.1, 0.25, 0.5, 1.0]
}

TRIALS = 500
STEPS_PER_TRIAL = 1000
RUNS_PER_CONFIG = 10

# Create results directory
RESULTS_DIR = Path("parameter_sweep_results")
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def create_model(tau, beta, lr):
    """Create model with given parameters"""
    return ParallelBetaIsingEdgeCLLN(
        HEIGHT=4,
        WIDTH=4,
        beta=beta,
        lr=lr,
        sigma=0.01,
        dt=1.0/tau  # Map tau to dt
    )

def run_single_experiment(params):
    """Run a single experiment configuration"""
    tau, beta, lr, eta, run_id, config_id = params
    
    # Create unique identifier for this run
    run_key = f"tau_{tau}_config_{config_id}_run_{run_id}"
    checkpoint_file = CHECKPOINT_DIR / f"{run_key}.pkl"
    
    # Initialize process-specific progress tracking
    process_id = os.getpid()
    desc = f"Config {config_id}, Run {run_id} (PID {process_id})"
    
    try:
        # Check if this run already completed
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    result = pickle.load(f)
                if result.get('completed', False):
                    return result
            except:
                pass  # Corrupted checkpoint, restart
        
        # Initialize or load from checkpoint
        start_trial = 0
        accuracy = []
        
        # Create model with error handling
        try:
            model = create_model(tau, beta, lr)
            if model is None:
                raise ValueError("create_model returned None")
        except Exception as e:
            error_msg = f"Model creation failed - Config {config_id}, Run {run_id} (PID {os.getpid()})"
            print(f"\n[MODEL ERROR] {error_msg}")
            print(f"Parameters: tau={tau}, beta={beta}, lr={lr}, eta={eta}")
            print(f"Error: {str(e)}\n")
            raise ValueError(f"Failed to create model: {e}")
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                if not checkpoint.get('completed', False):
                    start_trial = checkpoint.get('trial', 0)
                    accuracy = checkpoint.get('accuracy', [])
                    # Don't load the model from checkpoint to avoid serialization issues
                    # model = checkpoint.get('model')
            except Exception as e:
                print(f"[WARNING] Corrupted checkpoint for Config {config_id}, Run {run_id} - starting fresh: {str(e)}")
                pass  # Start fresh if checkpoint is corrupted
        
        # Run trials with progress tracking - create process-specific progress bar
        with tqdm(total=TRIALS, desc=desc, position=config_id % 8, leave=False, 
                  initial=start_trial, unit="trial", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for t in range(start_trial, TRIALS):
                # XOR learning logic
                index = random.choice([0, 1, 2, 3])
                label = 2 * (index > 0 and index < 3) - 1
                
                unclamped_data = unclamped[index].copy()
                
                converged = False
                count = 0
                step_desc = f"Trial {t+1}, Steps"
                
                # Inner progress bar for steps within each trial
                with tqdm(total=STEPS_PER_TRIAL, desc=step_desc, position=config_id % 8 + 8, 
                         leave=False, unit="step", disable=t % 100 != 0,  # Only show every 100th trial
                         bar_format='{desc}: {n_fmt}/{total_fmt} steps') as step_pbar:
                    
                    while not converged:
                        oplus = model.free_node_states[2, 2]
                        ominus = model.free_node_states[0, 0]
                        output = oplus - ominus
                        
                        # Apply nudging with eta parameter
                        oplusc = oplus + eta/2 * (label - output)
                        ominusc = ominus - eta/2 * (label - output)
                        
                        clamped_data = unclamped[index].copy()
                        clamped_data[2, 2] = oplusc
                        clamped_data[0, 0] = ominusc
                        
                        model.step(unclamped=unclamped_data, clamped=clamped_data)
                        model.train(1)
                        
                        count += 1
                        step_pbar.update(1)
                        
                        if count > STEPS_PER_TRIAL:
                            converged = True
                
                accuracy.append(output * label > 0)
                
                # Update progress bar with current accuracy
                recent_acc = np.mean(accuracy[-min(100, len(accuracy)):])
                pbar.set_postfix({
                    'acc': f'{recent_acc:.3f}',
                    'steps': count,
                    'tau': tau,
                    'beta': beta,
                    'lr': lr,
                    'eta': eta
                })
                pbar.update(1)
                
                # Save checkpoint every 100 trials
                if t % 100 == 0:
                    checkpoint = {
                        'params': {'tau': tau, 'beta': beta, 'lr': lr, 'eta': eta},
                        'trial': t + 1,
                        'accuracy': accuracy,
                        # 'model': model,  # Don't save model to avoid serialization issues
                        'completed': False
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint, f)
                    
                    # Update summary statistics and progress every 100 trials
                    try:
                        completed_count, error_count = update_summary_statistics()
                        create_progress_summary()
                        if t % 1000 == 0:  # Print update every 1000 trials to avoid spam
                            print(f"\n[PID {process_id}] Updated summary: {completed_count} completed, {error_count} errors")
                    except Exception as e:
                        print(f"[WARNING] Summary update failed for PID {process_id}: {str(e)}")
                        pass  # Don't let summary update errors break the experiment
        
        # Calculate final metrics
        final_accuracy = np.mean(accuracy[-1000:])  # Last 1000 trials
        convergence_point = None
        
        # Find convergence point (first time accuracy stays >90% for 100 trials)
        for i in range(100, len(accuracy)):
            if np.mean(accuracy[i-100:i]) > 0.9:
                convergence_point = i - 100
                break
        
        # Calculate weight statistics
        horiz_stats = {
            'mean': np.mean(model.horiz_weights),
            'std': np.std(model.horiz_weights),
            'max_abs': np.max(np.abs(model.horiz_weights))
        }
        verti_stats = {
            'mean': np.mean(model.verti_weights),
            'std': np.std(model.verti_weights),
            'max_abs': np.max(np.abs(model.verti_weights))
        }
        
        result = {
            'params': {'tau': tau, 'beta': beta, 'lr': lr, 'eta': eta},
            'run_id': run_id,
            'config_id': config_id,
            'final_accuracy': final_accuracy,
            'convergence_point': convergence_point,
            'accuracy_history': accuracy,
            'horiz_weight_stats': horiz_stats,
            'verti_weight_stats': verti_stats,
            'completed': True
        }
        
        # Save final result
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Print success message
        print(f"[SUCCESS] Config {config_id}, Run {run_id} (PID {os.getpid()}) completed - Final accuracy: {final_accuracy:.4f}")
        
        # Update summary statistics and progress when experiment completes
        try:
            update_summary_statistics()
            create_progress_summary()
        except Exception as e:
            print(f"[WARNING] Summary update failed after completion for Config {config_id}, Run {run_id}: {str(e)}")
            pass  # Don't let summary update errors break the experiment
        
        return result
        
    except Exception as e:
        # Print error immediately to terminal for real-time monitoring
        import traceback
        error_msg = f"ERROR in experiment - Config {config_id}, Run {run_id} (PID {os.getpid()})"
        print(f"\n{'='*60}")
        print(f"{error_msg}")
        print(f"Parameters: tau={tau}, beta={beta}, lr={lr}, eta={eta}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Trial when error occurred: {len(accuracy) if 'accuracy' in locals() else 0}")
        print(f"Traceback:")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        
        # Save error state for debugging with enhanced analytics
        error_result = {
            'params': {'tau': tau, 'beta': beta, 'lr': lr, 'eta': eta},
            'run_id': run_id,
            'config_id': config_id,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'completed': False,
            'trial': len(accuracy) if 'accuracy' in locals() else 0,
            'accuracy': accuracy if 'accuracy' in locals() else [],
            'process_id': os.getpid(),
            'timestamp': time.time()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(error_result, f)
        
        # Update summary statistics and progress when error occurs
        try:
            update_summary_statistics()
            create_progress_summary()
        except Exception as summary_error:
            print(f"[WARNING] Summary update failed after error for Config {config_id}, Run {run_id}: {str(summary_error)}")
            pass  # Don't let summary update errors break the experiment
        
        return error_result

def analyze_errors(results):
    """Analyze and report on experiment errors"""
    errors = [r for r in results if not r.get('completed', False)]
    
    if not errors:
        print("No errors encountered!")
        return
    
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {len(errors)}")
    
    # Group by error type
    error_types = {}
    for error in errors:
        error_type = error.get('error_type', 'Unknown')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    for error_type, error_list in error_types.items():
        print(f"{error_type}: {len(error_list)} occurrences")
        if error_list:
            print(f"  Example: {error_list[0]['error']}")
    
    # Analyze parameter patterns in errors
    print("\nError parameter patterns:")
    tau_errors = [e['params']['tau'] for e in errors]
    beta_errors = [e['params']['beta'] for e in errors]
    lr_errors = [e['params']['lr'] for e in errors]
    eta_errors = [e['params']['eta'] for e in errors]
    
    print(f"  Tau distribution: {set(tau_errors)}")
    print(f"  Beta distribution: {set(beta_errors)}")
    print(f"  LR distribution: {set(lr_errors)}")
    print(f"  Eta distribution: {set(eta_errors)}")
    print("=====================\n")

def print_progress_analytics(results_so_far, total_experiments):
    """Print analytics about the progress of experiments"""
    if not results_so_far:
        return
    
    completed = [r for r in results_so_far if r.get('completed', False)]
    errored = [r for r in results_so_far if not r.get('completed', False)]
    
    print(f"\n=== Progress Analytics ===")
    print(f"Completed: {len(completed)}/{total_experiments} ({len(completed)/total_experiments*100:.1f}%)")
    print(f"Errors: {len(errored)} ({len(errored)/len(results_so_far)*100:.1f}% of processed)")
    
    if completed:
        accuracies = [r['final_accuracy'] for r in completed]
        convergence_rates = [r['convergence_point'] is not None for r in completed]
        
        print(f"Accuracy stats: μ={np.mean(accuracies):.3f}, σ={np.std(accuracies):.3f}, "
              f"min={np.min(accuracies):.3f}, max={np.max(accuracies):.3f}")
        print(f"Convergence rate: {np.mean(convergence_rates)*100:.1f}%")
        
        # Best configuration so far
        best = max(completed, key=lambda x: x['final_accuracy'])
        print(f"Best so far: acc={best['final_accuracy']:.3f}, "
              f"tau={best['params']['tau']}, beta={best['params']['beta']}, "
              f"lr={best['params']['lr']}, eta={best['params']['eta']}")
    
    print("========================\n")

def create_progress_summary():
    """Create a lightweight progress summary for real-time monitoring"""
    try:
        # Count files by status
        completed_files = 0
        error_files = 0
        in_progress_files = 0
        total_files = 0
        
        # Quick file scan without full loading
        for checkpoint_file in CHECKPOINT_DIR.glob("*.pkl"):
            total_files += 1
            try:
                with open(checkpoint_file, 'rb') as f:
                    result = pickle.load(f)
                
                if result.get('completed', False):
                    completed_files += 1
                elif result.get('error', None):
                    error_files += 1
                else:
                    in_progress_files += 1
                    
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint file {checkpoint_file.name} for progress summary: {str(e)}")
                error_files += 1
        
        # Calculate total expected experiments
        total_expected = len(list(itertools.product(*PARAM_GRID.values()))) * RUNS_PER_CONFIG
        
        progress_summary = {
            'timestamp': time.time(),
            'last_updated': datetime.fromtimestamp(time.time()).isoformat(),
            'total_expected_experiments': total_expected,
            'experiments_started': total_files,
            'experiments_completed': completed_files,
            'experiments_in_progress': in_progress_files,
            'experiments_with_errors': error_files,
            'completion_percentage': (completed_files / total_expected) * 100 if total_expected > 0 else 0,
            'started_percentage': (total_files / total_expected) * 100 if total_expected > 0 else 0
        }
        
        # Save progress summary
        with open(RESULTS_DIR / "progress_summary.json", 'w') as f:
            json.dump(progress_summary, f, indent=2)
            
        return progress_summary
        
    except Exception as e:
        print(f"Error creating progress summary: {e}")
        return None

def update_summary_statistics():
    """Update summary statistics file with latest results"""
    try:
        results = []
        error_count = 0
        
        # Load all checkpoint files
        for checkpoint_file in CHECKPOINT_DIR.glob("*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    result = pickle.load(f)
                results.append(result)
                if not result.get('completed', False):
                    error_count += 1
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint file {checkpoint_file.name}: {str(e)}")
                error_count += 1
        
        # Filter completed results
        completed_results = [r for r in results if r.get('completed', False)]
        
        # Group by parameter configuration
        config_results = {}
        for result in completed_results:
            params = result['params']
            key = (params['tau'], params['beta'], params['lr'], params['eta'])
            if key not in config_results:
                config_results[key] = []
            config_results[key].append(result)
        
        # Aggregate statistics for each configuration
        summary_stats = []
        for config, runs in config_results.items():
            tau, beta, lr, eta = config
            
            final_accuracies = [r['final_accuracy'] for r in runs]
            convergence_points = [r['convergence_point'] for r in runs if r['convergence_point'] is not None]
            
            summary = {
                'tau': tau,
                'beta': beta,
                'lr': lr,
                'eta': eta,
                'n_runs': len(runs),
                'mean_final_accuracy': np.mean(final_accuracies),
                'std_final_accuracy': np.std(final_accuracies),
                'mean_convergence_time': np.mean(convergence_points) if convergence_points else None,
                'convergence_rate': len(convergence_points) / len(runs),
                'runs': runs
            }
            summary_stats.append(summary)
        
        # Save summary statistics with timestamp
        summary_with_metadata = {
            'timestamp': time.time(),
            'total_experiments': len(results),
            'completed_experiments': len(completed_results),
            'error_experiments': error_count,
            'summary_stats': summary_stats
        }
        
        # Save JSON version for easy reading
        with open(RESULTS_DIR / "summary_statistics.json", 'w') as f:
            json_stats = []
            for stat in summary_stats:
                json_stat = {}
                for key, value in stat.items():
                    if key != 'runs':  # Skip the detailed runs data for JSON
                        if isinstance(value, np.ndarray):
                            json_stat[key] = value.tolist()
                        elif isinstance(value, (np.int64, np.float64)):
                            json_stat[key] = value.item()
                        else:
                            json_stat[key] = value
                json_stats.append(json_stat)
            
            json_output = {
                'timestamp': summary_with_metadata['timestamp'],
                'last_updated': datetime.fromtimestamp(summary_with_metadata['timestamp']).isoformat(),
                'total_experiments': summary_with_metadata['total_experiments'],
                'completed_experiments': summary_with_metadata['completed_experiments'], 
                'error_experiments': summary_with_metadata['error_experiments'],
                'summary_stats': json_stats
            }
            json.dump(json_output, f, indent=2)
        
        # Save detailed pickle version
        with open(RESULTS_DIR / "detailed_results.pkl", 'wb') as f:
            pickle.dump(summary_with_metadata, f)
        
        return len(completed_results), error_count
        
    except Exception as e:
        print(f"Error updating summary statistics: {e}")
        return 0, 0

def collect_and_analyze_results():
    """Collect all results and perform analysis"""
    
    results = []
    error_count = 0
    
    # Load all checkpoint files
    for checkpoint_file in CHECKPOINT_DIR.glob("*.pkl"):
        try:
            with open(checkpoint_file, 'rb') as f:
                result = pickle.load(f)
            results.append(result)
            if not result.get('completed', False):
                error_count += 1
        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
            error_count += 1
    
    # Filter completed results
    completed_results = [r for r in results if r.get('completed', False)]
    
    print(f"Loaded {len(completed_results)} completed results, {error_count} errors/incomplete")
    
    # Group by parameter configuration
    config_results = {}
    for result in completed_results:
        params = result['params']
        key = (params['tau'], params['beta'], params['lr'], params['eta'])
        if key not in config_results:
            config_results[key] = []
        config_results[key].append(result)
    
    # Aggregate statistics for each configuration
    summary_stats = []
    for config, runs in config_results.items():
        tau, beta, lr, eta = config
        
        final_accuracies = [r['final_accuracy'] for r in runs]
        convergence_points = [r['convergence_point'] for r in runs if r['convergence_point'] is not None]
        
        summary = {
            'tau': tau,
            'beta': beta,
            'lr': lr,
            'eta': eta,
            'n_runs': len(runs),
            'mean_final_accuracy': np.mean(final_accuracies),
            'std_final_accuracy': np.std(final_accuracies),
            'mean_convergence_time': np.mean(convergence_points) if convergence_points else None,
            'convergence_rate': len(convergence_points) / len(runs),
            'runs': runs
        }
        summary_stats.append(summary)
    
    # Save summary statistics
    with open(RESULTS_DIR / "summary_statistics.json", 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_stats = []
        for stat in summary_stats:
            json_stat = {}
            for key, value in stat.items():
                if key != 'runs':  # Skip the detailed runs data for JSON
                    if isinstance(value, np.ndarray):
                        json_stat[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.float64)):
                        json_stat[key] = value.item()
                    else:
                        json_stat[key] = value
            json_stats.append(json_stat)
        json.dump(json_stats, f, indent=2)
    
    # Save detailed results
    with open(RESULTS_DIR / "detailed_results.pkl", 'wb') as f:
        pickle.dump(summary_stats, f)
    
    return summary_stats

def main():
    """Main execution function"""
    
    print("Starting parameter sweep experiment...")
    print(f"Total configurations: {len(list(itertools.product(*PARAM_GRID.values())))}")
    print(f"Runs per configuration: {RUNS_PER_CONFIG}")
    print(f"Total experiments: {len(list(itertools.product(*PARAM_GRID.values()))) * RUNS_PER_CONFIG}")
    
    # Create initial progress summary
    try:
        create_progress_summary()
        print("Created initial progress summary file")
    except Exception as e:
        print(f"Warning: Could not create initial progress summary: {e}")
    
    # Generate all parameter combinations
    param_combinations = []
    config_id = 0
    
    for tau, beta, lr, eta in itertools.product(*PARAM_GRID.values()):
        for run_id in range(RUNS_PER_CONFIG):
            param_combinations.append((tau, beta, lr, eta, run_id, config_id))
        config_id += 1
    
    # Determine number of processes to use
    n_processes = min(mp.cpu_count()//2, len(param_combinations))
    print(f"Using {n_processes} processes")
    
    # Run experiments in parallel with enhanced progress tracking
    start_time = time.time()
    
    print("Starting parallel execution with detailed progress tracking...")
    print(f"Each process will show individual progress bars")
    print(f"Position staggering: processes will use positions 0-7 and 8-15 for step tracking")
    
    with mp.Pool(n_processes) as pool:
        # Enhanced progress tracking with analytics
        results = []
        
        # Use imap_unordered to get results as they complete
        result_iter = pool.imap_unordered(run_single_experiment, param_combinations)
        
        # Create main progress bar
        with tqdm(total=len(param_combinations), desc="Overall Progress", position=0, 
                  unit="experiment", leave=True,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as main_pbar:
            
            for i, result in enumerate(result_iter):
                results.append(result)
                main_pbar.update(1)
                
                # Update progress bar with current statistics
                completed_count = len([r for r in results if r.get('completed', False)])
                error_count = len([r for r in results if not r.get('completed', False)])
                
                if completed_count > 0:
                    completed_results = [r for r in results if r.get('completed', False)]
                    avg_acc = np.mean([r['final_accuracy'] for r in completed_results])
                    main_pbar.set_postfix({
                        'completed': completed_count,
                        'errors': error_count,
                        'avg_acc': f'{avg_acc:.3f}'
                    })
                
                # Print detailed analytics every 50 completed experiments
                if (i + 1) % 50 == 0:
                    print_progress_analytics(results, len(param_combinations))
    
    elapsed_time = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed_time/3600:.2f} hours")
    
    # Final progress analytics
    print_progress_analytics(results, len(param_combinations))
    
    # Error analysis
    analyze_errors(results)
    
    # Additional execution statistics
    total_completed = len([r for r in results if r.get('completed', False)])
    total_errors = len([r for r in results if not r.get('completed', False)])
    
    print(f"=== Final Execution Summary ===")
    print(f"Total experiments: {len(param_combinations)}")
    print(f"Successfully completed: {total_completed}")
    print(f"Failed/Errored: {total_errors}")
    print(f"Success rate: {total_completed/len(param_combinations)*100:.1f}%")
    print(f"Average time per experiment: {elapsed_time/len(param_combinations):.2f} seconds")
    print(f"Throughput: {len(param_combinations)/elapsed_time:.2f} experiments/second")
    print("==============================\n")
    
    # Collect and analyze results
    print("Analyzing results...")
    summary_stats = collect_and_analyze_results()
    
    # Print summary
    successful_configs = len([s for s in summary_stats if s['n_runs'] == RUNS_PER_CONFIG])
    print(f"Successfully completed {successful_configs} parameter configurations")
    
    # Find best performing configuration
    best_config = max(summary_stats, key=lambda x: x['mean_final_accuracy'])
    print(f"Best configuration: tau={best_config['tau']}, beta={best_config['beta']}, "
          f"lr={best_config['lr']}, eta={best_config['eta']}")
    print(f"Best accuracy: {best_config['mean_final_accuracy']:.4f} ± {best_config['std_final_accuracy']:.4f}")
    
    return summary_stats

if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    mp.set_start_method('spawn', force=True)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    summary_stats = main()