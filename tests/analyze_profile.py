import pstats

p = pstats.Stats("profile.stats")
print("--- Top 20 by Cumulative Time ---")
p.sort_stats("cumtime").print_stats(20)

print("\n--- Top 20 by Total Time (Self Time) ---")
p.sort_stats("tottime").print_stats(20)
