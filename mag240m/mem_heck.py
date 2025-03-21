import tracemalloc

# Load the snapshot file
snapshot = tracemalloc.Snapshot.load("tracemalloc_snapshot.dump")

# Get memory allocation statistics
top_stats = snapshot.statistics("lineno")

# Print the top 10 memory allocations
print("[ Top 10 memory allocations ]")
for stat in top_stats[:10]:
    print(stat)
