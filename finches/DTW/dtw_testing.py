import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

from constraints import create_sakoe_chiba_mask
from dtw_stateless import calculate_dtw_distance_flex, find_dtw_path_flex, calculate_barycenter_average, softdtw_barycenter



if __name__ == '__main__':
    # --- 1. Define sample data ---
    # simple different max value example
    # s1 = np.array([1., 2., 3., 4., 5., 5., 5.])
    # s2 = np.array([1., 2., 3., 4., 4., 4., 4.])

    # simple sine wave with a shift example
    t = np.linspace(0, 2*np.pi, 500)
    s1 = np.sin(t)
    s2 = np.sin(t+0.3)

    print("--- 2. Standard DTW (symmetric1, no global constraint) ---")
    dist_std, acc_std = calculate_dtw_distance_flex(s1, s2, step_pattern='symmetric1')
    print(f"Standard DTW Distance: {dist_std:.4f}")

    print("\n--- 3. DTW with a Sakoe-Chiba global constraint ---")
    # Create a global mask with a window of 1
    sc_mask = create_sakoe_chiba_mask(len(s1), len(s2), window=1)
    dist_sc, acc_sc = calculate_dtw_distance_flex(s1, s2, global_mask=sc_mask, step_pattern='symmetric1')
    print(f"DTW Distance with Sakoe-Chiba (window=1): {dist_sc:.4f}")

    print("\n--- 4. DTW with a different local step pattern ('symmetric2') ---")
    # This pattern penalizes non-diagonal moves more heavily.
    dist_s2, acc_s2 = calculate_dtw_distance_flex(s1, s2, step_pattern='symmetric2')
    print(f"DTW Distance with 'symmetric2' step pattern: {dist_s2:.4f}")
    
    # --- 5. Find and analyze a path ---
    print("\n--- Path Finding and Analysis ---")
    path_s2 = find_dtw_path_flex(acc_s2, step_pattern='symmetric2')
    print(f"Path with 'symmetric2' has {len(path_s2)} steps.")
    print("Path starts at", path_s2[0], "and ends at", path_s2[-1])

    # --- 6. (Optional) Visualization ---
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        def plot_alignment(ax, acc_cost, path, title):
            ax.imshow(acc_cost.T, origin='lower', cmap='gray_r', interpolation='nearest')
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-')
            ax.set_title(title)
            ax.set_xlabel("s1 index")
            ax.set_ylabel("s2 index")
        
        path_std = find_dtw_path_flex(acc_std)
        path_sc = find_dtw_path_flex(acc_sc)

        plot_alignment(axs[0], acc_std, path_std, f"Standard DTW\nDist: {dist_std:.2f}")
        plot_alignment(axs[1], acc_sc, path_sc, f"Sakoe-Chiba (w=1)\nDist: {dist_sc:.2f}")
        plot_alignment(axs[2], acc_s2, path_s2, f"Step Pattern 'symmetric2'\nDist: {dist_s2:.2f}")
        
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")




print(f"\n\n\n Showing off the DTW Barycenter Averaging (DBA)\n\n\n")






if __name__ == '__main__':
    # 1. Create sample signals of different lengths
    # They are all sine waves with different lengths, phase shifts, and noise
    t1 = np.linspace(0, 2 * np.pi, 80)
    s1 = np.sin(t1) + np.random.rand(len(t1)) * 0.1

    t2 = np.linspace(0.5, 2.5 * np.pi, 100) # Different length and phase
    s2 = np.sin(t2) + np.random.rand(len(t2)) * 0.1

    t3 = np.linspace(-0.5, 1.5 * np.pi, 70) # Different length and phase
    s3 = np.sin(t3) + np.random.rand(len(t3)) * 0.1

    sequences = [s1, s2, s3]

    # 2. Calculate the barycenter average
    # We will use the first sequence (s1) to determine the length of the average.
    average_sequence = calculate_barycenter_average(
        sequences,
        n_iterations=5, # Fewer iterations for a quick example
        initial_center_index=1
    )

    # 3. Plot the results for visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Plot original signals with some transparency
    plt.plot(s1, "o-", label="Signal 1 (length 80)", alpha=0.5)
    plt.plot(s2, "x-", label="Signal 2 (length 100)", alpha=0.5)
    plt.plot(s3, "s-", label="Signal 3 (length 70)", alpha=0.5)

    # Plot the computed barycenter average with a thicker, solid line
    if average_sequence is not None:
        plt.plot(
            average_sequence, "k-", 
            label=f"Barycenter Average (length {len(average_sequence)})", 
            linewidth=3
        )

    plt.title("DTW Barycenter Averaging of Signals with Different Lengths", fontsize=16)
    plt.xlabel("Time (Index)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()










    print(f"\n\n\n\nSoft Gradient version for true optimization\n\n\n\n")







    # --- Main Example ---

if __name__ == '__main__':
    # 1. Create the same sample signals of different lengths
    t1 = np.linspace(0, 2 * np.pi, 80)
    s1 = np.sin(t1) + np.random.rand(len(t1)) * 0.1
    t2 = np.linspace(0.5, 2.5 * np.pi, 100)
    s2 = np.sin(t2) + np.random.rand(len(t2)) * 0.1
    t3 = np.linspace(-0.5, 1.5 * np.pi, 70)
    s3 = np.sin(t3) + np.random.rand(len(t3)) * 0.1
    sequences = [s1, s2, s3]

    # 2. Calculate the barycenter using BOTH methods for comparison
    
    # --- Classic DBA ---
    print("--- Calculating Barycenter with classic DBA ---")
    dba_average = calculate_barycenter_average(sequences, n_iterations=10, initial_center_index=0)
    
    print("\n--- Calculating Barycenter with Soft-DTW ---")
    # Soft-DTW parameters
    # Gamma controls the smoothness. A small value makes it closer to hard DTW.
    # Learning rate controls the step size of the optimization.
    softdtw_average = softdtw_barycenter(
        sequences, 
        gamma=0.1, 
        n_iterations=50, 
        learning_rate=0.05
    )
    
    # 3. Plot the results for visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    # Plot original signals
    plt.plot(s1, "o-", color='skyblue', label="Signal 1", alpha=0.6)
    plt.plot(s2, "x-", color='salmon', label="Signal 2", alpha=0.6)
    plt.plot(s3, "s-", color='lightgreen', label="Signal 3", alpha=0.6)

    # Plot the DBA average
    plt.plot(dba_average, "-", color='blue', label="DBA Barycenter (Heuristic)", linewidth=3, alpha = 0.3)
    
    # Plot the Soft-DTW average
    plt.plot(softdtw_average, "--", color='red', label="Soft-DTW Barycenter (Gradient-based)", linewidth=3)

    plt.title("Comparing DBA and Soft-DTW Barycenters", fontsize=16)
    plt.xlabel("Time (Index)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
