import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import deque


class VCNetworkContagion:
    def __init__(self, num_nodes=1000, edges_per_node=3):
        # Create a scale-free network to simulate the VC/Startup ecosystem
        self.num_nodes = num_nodes
        self.graph = nx.barabasi_albert_graph(n=num_nodes, m=edges_per_node)

        # Initialize financial 'stress' and a failure 'threshold' for each node
        # Hubs (highly connected nodes) have higher thresholds to simulate larger capital reserves
        self.stress = {node: 0 for node in self.graph.nodes()}
        self.threshold = {node: max(3, self.graph.degree(node)) for node in self.graph.nodes()}
        self.avalanche_sizes = []

    def inject_noise(self):
        # Randomly apply financial stress (cash burn / market shock) to one node
        target = np.random.choice(list(self.graph.nodes()))
        self.stress[target] += 1

        # Check if this triggers a failure
        if self.stress[target] >= self.threshold[target]:
            self._trigger_avalanche(target)

    def _trigger_avalanche(self, start_node):
        # Using a queue to resolve the cascade across the network
        queue = deque([start_node])
        current_avalanche_size = 0
        toppled_this_round = {start_node}

        while queue:
            current_node = queue.popleft()

            # If the node fails, it resets its stress but distributes it to connected partners
            if self.stress[current_node] >= self.threshold[current_node]:
                self.stress[current_node] = 0
                current_avalanche_size += 1

                # Distribute stress to neighbors (connectivity)
                neighbors = list(self.graph.neighbors(current_node))
                for neighbor in neighbors:
                    self.stress[neighbor] += 1
                    # If neighbor crosses threshold and hasn't toppled yet in this chain, add to queue
                    if self.stress[neighbor] >= self.threshold[neighbor] and neighbor not in toppled_this_round:
                        queue.append(neighbor)
                        toppled_this_round.add(neighbor)

        if current_avalanche_size > 0:
            self.avalanche_sizes.append(current_avalanche_size)

    def run(self, steps=25000):
        print(f"Simulating {steps} market events across {self.num_nodes} interconnected entities...")
        for _ in range(steps):
            self.inject_noise()

    def analyze_and_plot(self):
        if not self.avalanche_sizes:
            print("No cascades recorded.")
            return

        # Calculate frequencies
        sizes = np.array(self.avalanche_sizes)
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        probabilities = counts / len(sizes)

        # Plotting the log-log distribution
        plt.figure(figsize=(10, 6))
        plt.scatter(unique_sizes, probabilities, color="#e74c3c", alpha=0.7, edgecolors="black", label="Simulated Data")

        # Fit a power law line (y = A * x^(-alpha)) for the linear region on log-log
        # We take log(y) = log(A) - alpha * log(x)
        log_x = np.log10(unique_sizes)
        log_y = np.log10(probabilities)

        # Simple linear fit on log-log data
        def linear_func(x, m, c):
            return m * x + c

        try:
            popt, _ = curve_fit(linear_func, log_x, log_y)
            alpha, log_A = -popt[0], popt[1]
            fit_y = 10 ** (linear_func(log_x, popt[0], popt[1]))

            plt.plot(
                unique_sizes,
                fit_y,
                color="black",
                linestyle="--",
                label=f"Power Law Fit ($\\alpha$ = {alpha:.2f})",
            )
            print(f"Calculated Power Law Exponent (alpha): {alpha:.2f}")
        except Exception:
            print("Could not fit power law line.")

        plt.xscale("log")
        plt.yscale("log")
        plt.title("Self-Organized Criticality in VC Networks")
        plt.xlabel("Avalanche Size (Number of Bankruptcies)")
        plt.ylabel("Probability P(S)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)

        plt.savefig("complex_network_avalanche.png", dpi=300)
        print("Analysis complete. High-res plot saved as 'complex_network_avalanche.png'.")
        plt.show()


# --- Execution ---
if __name__ == "__main__":
    # 1000 nodes, each connecting to 2 existing nodes upon creation (classic Scale-Free setup)
    model = VCNetworkContagion(num_nodes=1000, edges_per_node=2)
    model.run(steps=35000)
    model.analyze_and_plot()
