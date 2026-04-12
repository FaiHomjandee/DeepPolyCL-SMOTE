 #!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8

# File: 1_grid_search_selfCLSMOTE.py
import itertools
import pandas as pd
import torch
import time
from collections import Counter
from pathlib import Path # Import Path
import numpy as np 


import setproctitle

# Import the tools you need from your utils file
from defs_CoSMO import run_cosmo_supcon,run_l2_smote_variants

data = 'cifar10'
DATA_DIR = Path("/home/ubuntu/fai_run/project/prepared_data")

if __name__ == "__main__":
    # --- 0. Load the pre-generated data from the files ---
    print("Loading pre-generated data from files...")
    try:
        # --- Construct the full, correct file paths ---
        folds_data_path = DATA_DIR / f"{data}_folds_data.pth"
        imbalanced_data_path = DATA_DIR / f"{data}_imbalanced_data.pth"

        print(f"Attempting to load: {folds_data_path}")
        print(f"Attempting to load: {imbalanced_data_path}")

        # --- Load from the corrected paths ---
        folds_data = torch.load(folds_data_path, weights_only=False)
        imbalanced_train_dataset_list = torch.load(imbalanced_data_path, weights_only=False)
        
        print("\nData loaded successfully.")
        
    except FileNotFoundError:
        print("\nERROR: Data files not found! Please check the following:")
        print("1. Make sure you have successfully run the 'prepare_data.py' script.")
        print(f"2. Check that the files exist at this exact location: {DATA_DIR}")
        exit()
    
    # --- 1. Load the data ONCE ---
    first_fold_data_loaders = folds_data[0]
    first_fold_imbalanced_list = imbalanced_train_dataset_list[0]

    train_loader_ae, val_loader, test_loader = first_fold_data_loaders
    
    # For the training set
    all_train_labels = []
    for _, labels_batch in train_loader_ae:
        all_train_labels.extend(labels_batch.cpu().tolist())
    train_distribution = Counter(all_train_labels)
    print('Train Dist:', train_distribution)

    # For the validation set
    all_val_labels = []
    for _, labels_batch in val_loader:
        all_val_labels.extend(labels_batch.cpu().tolist())
    val_distribution = Counter(all_val_labels)
    print('Val Dist:', val_distribution)

    # For the test set
    all_test_labels = []
    for _, labels_batch in test_loader:
        all_test_labels.extend(labels_batch.cpu().tolist())
    test_distribution = Counter(all_test_labels)
    print('Test Dist:', test_distribution)
        
    # --- 1. Load the data ONCE ---
    #folds_data, imbalanced_train_dataset_list = load_all_data()
    first_fold_data_loaders = folds_data[0]
    first_fold_imbalanced_list = imbalanced_train_dataset_list[0]

    # # --- 2. Define the grid for THIS model ---
    method = 'CoSMO'
    param_grid = {
                'lr': [2e-4],
                'alpha': [0.9],
                'temperature': [0.07],
                'n_z': [600],
                'weight_decay':[1e-4],
                'dataset_name': [data]
    } 
    
    # Default
    # param_grid = {
    #     lr': [0.0002, 0.001],
    #     alpha': [0.7, 0.5, 0.3],
    #     temperature': [0.07, 0.1],
    #     n_z': [300],
    #     weight_decay':[0.0005, 0.001]
    #     }
    
    # --- 2. DeepSMOTE ---
    # method = 'DeepSMOTE'
    # param_grid = {
    #     'lr': [0.0002],   
    #     'n_z': [600],#20], #[600], 
    #     'dataset_name': [data]
    #     # Latent comparison animation
    #     # 'save_latent_anim': [True],
    #     # 'anim_separate': [False],
    #     # 'save_latent_compare_anim': [True],
    #     # 'latent_compare_method': ['pca_tsne'],  # 'pca', 'tsne', 'pca_tsne','umap'
    # }
    
    # --- 2. BAGAN ---
    # method = 'BAGAN'
    # param_grid = {
    #     'lr_g': [2e-4],   
    #     'lr_d': [2e-4], 
    #     'n_z': [600], 
    #     'dataset_name':[data]
    # }
    
    # # --- 2. GAMO ---
    # method = 'GAMO'
    # param_grid = {
    #         'lr_fc': [0.001],
    #         'lr_gan': [0.0005],
    #         'n_z': [600],
    #         'freeze_epochs': [80],
    #         'warmup_epochs': [20],
    #         'dataset_name': [data]      
    # }
    
    # ----- l2 --------
    # param_grid = {
    #     'lr': [0.0002],   
    #     'n_z': [300], 
    #     'weight_decay':[1e-3] 
    # }
    
    # # ----- plain ----
    # method = 'Baseline'
    # param_grid = {
    #     'dataset_name': [data]      
    # }
    
    
    # model = {'DeepSMOTE','DeepSMOTE','BAGAN','GAMO'}
    path = '/home/ubuntu/fai_run/project/'

    results_filepath1_resnet = f"{path}Results_{method}_avg_{data}_size32_avg50_IMG.csv"
    setproctitle.setproctitle(results_filepath1_resnet) # Change this to your desired tag

    results_data1,results_data2,results_data3 = [],[],[]
    
    results_list = []
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"--- Starting Grid Search ---")
    
    # num_folds = len(folds_data)
    # num_folds = 1 
    num_folds = 3
    # --- 3. Loop through experiments ---
    for i, params in enumerate(experiments):
        print(f"\n--- RUN {i+1}/{len(experiments)} | PARAMS: {params} ---")
        
        fold_f1_scores = []
        fold_asca_scores = []
        fold_gm_scores = []
        fold_perclass_reports = [] # To store all reports
        
        print(f"Running {num_folds} folds for this parameter set...")
        
        for fold_idx in range(num_folds):
            # fold_idx = 2            
            print(f"    --- Fold {fold_idx + 1}/{num_folds} ---")
            # Get the data for the *current* fold
            current_fold_loaders = folds_data[fold_idx]
            current_fold_imbalanced_list = imbalanced_train_dataset_list[fold_idx] #run_single_fold_selfCLSMOTE_selfsup_warmup

            f1,asca,g_mean,per_class_results = run_single_fold_selfCLSMOTE_selfsup_warmup(
                                                params, 
                                                current_fold_loaders, 
                                                current_fold_imbalanced_list
                                                # fold_idx
                                            )

            # f1,asca,g_mean,per_class_results = run_single_fold_imbalanced_baseline(
            #                                     params,
            #                                     current_fold_loaders, 
            #                                     current_fold_imbalanced_list
            #                                 )
            
            fold_f1_scores.append(f1)
            fold_asca_scores.append(asca)
            fold_gm_scores.append(g_mean)
            fold_perclass_reports.append(per_class_results) # Store the report 
            
            # --- ✅ 1. Store individual fold result ---
            fold_result_row = params.copy()
            fold_result_row['Fold'] = fold_idx + 1 # Add fold number
            fold_result_row['F1'] = f1
            fold_result_row['ASCA'] = asca
            fold_result_row['GM'] = g_mean
            fold_result_row['Perclass_Report'] = per_class_results
            results_list.append(fold_result_row)
            
            # --- ✅ 2. SAVE AFTER EVERY FOLD ---
            # The CSV will be created/updated right after Fold 1.
            df = pd.DataFrame(results_list)
            df.to_csv(results_filepath1_resnet, index=False)
            print(f"    ... Saved results for fold {fold_idx + 1}")
        
        # --- Calculate Averages ---
        avg_f1 = np.mean(fold_f1_scores)
        avg_asca = np.mean(fold_asca_scores)
        avg_gm = np.mean(fold_gm_scores)
        
        std_f1 = np.std(fold_f1_scores)
        std_asca = np.std(fold_asca_scores)
        std_gm = np.std(fold_gm_scores)
        
        print(f"  --- Average Results for {params} ---")
        print(f"  Avg F1:   {avg_f1:.4f} (±{std_f1:.4f})")
        print(f"  Avg ASCA: {avg_asca:.4f} (±{std_asca:.4f})")
        print(f"  Avg GM:   {avg_gm:.4f} (±{std_gm:.4f})")
        
        # --- ✅ 2. Store average result row ---
        avg_result_row = params.copy()
        avg_result_row['Fold'] = 'Average' # Use a string to identify this row
        avg_result_row['F1'] = avg_f1
        avg_result_row['ASCA'] = avg_asca
        avg_result_row['GM'] = avg_gm
        avg_result_row['Std_F1'] = std_f1
        avg_result_row['Std_ASCA'] = std_asca
        avg_result_row['Std_GM'] = std_gm
        avg_result_row['Perclass_Report'] = str(fold_perclass_reports) # Store all reports
        results_list.append(avg_result_row)
        
        # --- ✅ 4. SAVE AGAIN (to include the average row) ---
        df = pd.DataFrame(results_list)
        df.to_csv(results_filepath1_resnet, index=False)
        print(f"  ... Saved average results for this experiment")
        
        
    # --- 4. Find and print the best parameters ---
    print(f"\nGrid search complete. Results saved to: {results_filepath1_resnet}") 
    
    # Load the final CSV
    df_results = pd.read_csv(results_filepath1_resnet)
    
    # --- ✅ 4. Find best run from 'Average' rows ---
    # First, filter the DataFrame to only include the 'Average' rows
    avg_rows_df = df_results[df_results['Fold'] == 'Average'].copy()
    
    # Now, find the best run based on the 'F1' column (which holds avg F1)
    # Check if avg_rows_df is empty, in case the run was interrupted
    if not avg_rows_df.empty:
        best_run_avg = avg_rows_df.loc[avg_rows_df['F1'].idxmax()]
        print("\nBest Hyperparameters found (based on Avg_F1):")
        print(best_run_avg)
    else:
        print("\nNo 'Average' rows found. Could not determine best parameters.")

    
    time_end = time.time()
    print(f'\nTime spend (hr): {(time_end-time_start)/(60*60):.2f}')

