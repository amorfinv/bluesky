import random

# Define the range for the random seed values
seed_range_min = 0
seed_range_max = 2**32 - 1  # You can adjust this range based on your requirements

# Number of random seeds to generate
num_seeds = 15

# Generate unique random seeds
random_seeds = random.sample(range(seed_range_min, seed_range_max), num_seeds)

# Print the random seeds
print("Randomly selected seeds:")
for seed in random_seeds:
    print(seed)