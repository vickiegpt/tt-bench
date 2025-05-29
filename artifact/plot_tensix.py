import matplotlib.pyplot as plt

# Data for latency vs memory footprint
footprints_kb = [4, 16, 64, 256, 1024, 4096, 16384, 65536]
latencies = [0.375, 0.104639, 0.026178, 0.00584564, 0.00174484, 0.000385284, 9.36031e-05, 2.04444e-05]

# Data for bandwidth vs memory footprint
footprints_bw_kb = [64, 256, 1024, 4096, 16384, 65536]
bandwidths = [1098.86, 3869.01, 8922.97, 12230.1, 10022.1, 7710.43]

fig, ax1 = plt.subplots()

# Plot latency
ax1.plot(footprints_kb, latencies, marker='o', label='Latency (μs/element)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Memory Footprint (KB)')
ax1.set_ylabel('Latency (μs per element)')
ax1.tick_params(axis='y')

# Plot bandwidth on secondary axis
ax2 = ax1.twinx()
ax2.plot(footprints_bw_kb, bandwidths, marker='x', label='Bandwidth (MB/s)')
ax2.set_yscale('log')
ax2.set_ylabel('Bandwidth (MB/s)')
ax2.tick_params(axis='y')

# Title and legends
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('tensix_latency_bandwidth.pdf')