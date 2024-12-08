import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from statistics import mean, stdev
import time
from sympy import nextprime
import heapq



def generate_hash(num_hashes, p):
    """
    Generate a list of hash functions for MinHashing.

    Each hash function is of the form:
        h(x) = (a * x + b) % p
    where:
    - a and b are random integers chosen for each hash function.
    - p is a prime number larger than the maximum number of rows.

    Parameters:
        num_hashes (int): The number of hash functions to generate.
        p (int): A prime number to use as the modulus.

    Returns:
        list: A list of hash functions.
    """    
    random.seed(42)  # Ensure reproducibility of random numbers
    hash_funcs = []
    
    for _ in range(num_hashes):
        # Generate random coefficients for the hash function
        a = random.randint(1, p - 1)  # a should be non-zero
        b = random.randint(0, p - 1)  # b can be zero
        
        # Define a hash function and append it to the list
        hash_funcs.append(lambda x, a=a, b=b, p=p: (a * x + b) % p)
    
    return hash_funcs



def generate_permutations(num_permutations, max_rows):
    """
    Generate a list of true permutation functions.

    Parameters:
        num_permutations (int): Number of permutations to generate.
        max_rows (int): Number of rows to permute (from 0 to max_rows - 1).

    Returns:
        list: A list of permutation functions.
    """
    random.seed(42)  # Ensure reproducibility of random numbers
    permutations = []

    for _ in range(num_permutations):
        # Generate a random permutation of integers [0, max_rows - 1]
        perm = list(range(max_rows))
        random.shuffle(perm)

        # Create a function representing this permutation
        permutations.append(lambda x, perm=perm: perm[x])

    return permutations



def jaccard_error(user_movies, hash_funcs_list, num_samples=1000):
    """
    Compute the error between real Jaccard similarity and MinHash approximated similarity
    for varying numbers of hash functions and plot the results, including computation time.

    Parameters:
        user_movies (dict): Dictionary where keys are user IDs and values are sets of movies.
        hash_funcs_list (list of lists): A list of hash function lists, each representing a set of hash functions.
        num_samples (int): Number of random user pairs to sample.
    """
    # Step 1: Randomly sample pairs of users
    random.seed(42)
    user_ids = list(user_movies.keys())
    sampled_users = random.sample(user_ids, 2 * num_samples)
    sampled_pairs = [(sampled_users[i], sampled_users[i + 1]) for i in range(0, len(sampled_users), 2)]

    def minhash_signature(movies, hash_funcs):
        """
        Compute the MinHash signature for a set of movies using the provided hash functions.
        """
        return [min(hash_func(movie) for movie in movies) for hash_func in hash_funcs]

    avg_errors = []
    std_errors = []
    num_hashes = []
    runtimes = []

    # Step 2: Compute errors for each list of hash functions
    for hash_funcs in hash_funcs_list:
        num_hash = len(hash_funcs)
        num_hashes.append(num_hash)
        errors = []

        start_time = time.time()

        for u1, u2 in tqdm(sampled_pairs, desc=f"Processing users with k={num_hash}", total=num_samples):
            # Real Jaccard similarity
            set1, set2 = user_movies[u1], user_movies[u2]
            real_jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

            # MinHash signatures
            sig1 = minhash_signature(set1, hash_funcs)
            sig2 = minhash_signature(set2, hash_funcs)

            # Approximated Jaccard similarity using MinHash
            approx_jaccard = np.mean(np.array(sig1) == np.array(sig2))

            # Compute the error
            errors.append(abs(real_jaccard - approx_jaccard))

        end_time = time.time()
        runtimes.append(end_time - start_time)

        # Calculate error statistics for this number of functions
        avg_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))

    # Step 3: Plot the results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Number of Functions')
    ax1.set_ylabel('Error', color='blue')
    ax1.errorbar(num_hashes, avg_errors, yerr=std_errors, fmt='o-', capsize=5, capthick=2, color='blue', label='Avg Error ± Std')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Runtime (s)', color='red')
    ax2.plot(num_hashes, runtimes, 's-', color='red', label='Runtime')
    ax2.tick_params(axis='y', labelcolor='red')

    # Adjust legend placement (in the center, closer to the plot)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2)

    fig.suptitle('Error and Runtime vs Number of Functions')
    fig.tight_layout()
    plt.show()



def collision_probability(bands, rows_per_band):
    """
    Plots the probability of collision as a function of Jaccard similarity
    based on adjustable LSH parameters (number of bands and rows per band).
    
    This function calculates the probability of collision for varying Jaccard 
    similarities and plots the resulting curve. It also marks the Jaccard similarity 
    where the probability of collision is 0.5, which represents the probabilistic threshold.
    
    Parameters:
        bands (int): Number of bands used in LSH.
        rows_per_band (int): Number of rows per band in LSH.

    Returns:
        None: Displays the plot showing the relationship between Jaccard similarity 
              and the probability of collision, along with the probabilistic threshold.
    """
    # Generate a range of Jaccard similarity values from 0 to 1
    jaccard_similarities = np.linspace(0, 1, 500)

    # Calculate collision probabilities
    collision_probabilities = [
        1 - (1 - s**rows_per_band)**bands for s in jaccard_similarities
    ]

    # Find the probabilistic collision threshold (Jaccard similarity at P = 0.5)
    threshold_idx = np.abs(np.array(collision_probabilities) - 0.5).argmin()
    threshold_similarity = jaccard_similarities[threshold_idx]

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.plot(
        jaccard_similarities,
        collision_probabilities,
        label=f"{bands} bands, {rows_per_band} rows per band",
    )
    plt.axhline(0.5, color="red", linestyle="--", label="P = 0.5 (Threshold)")
    plt.axvline(
        threshold_similarity,
        color="blue",
        linestyle="--",
        label=f"Threshold (Jaccard ≈ {threshold_similarity:.2f})",
    )
    plt.xlim(0, 1)
    plt.title("Probability of Collision vs. Jaccard Similarity")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Probability of Collision")
    plt.grid(True)
    plt.legend()
    plt.show()



class LSH:

    def __init__(self, user_movies, bands, rows_per_band):
        """
        Initialize the LSH instance with user data, hashing parameters, and LSH structure.

        Args:
            user_movies (dict): A dictionary mapping user IDs to sets of movie IDs they have rated.
            bands (int): Number of bands to split the signature matrix into.
            rows_per_band (int): Number of rows per band within the signature matrix.
        """
        # Calculate the Jaccard similarity threshold for a collision probability of 1/2
        self.threshold = (1 - (1 / 2) ** (1 / bands)) ** (1 / rows_per_band)

        print(f"Initializing LSH with a target threshold of {self.threshold:.2f}...\n")

        # Find the smallest prime number greater than the maximum movie ID in the dataset
        # This prime is used to define the modular space for the hash functions
        self.p = nextprime(max(movie_id for movies in user_movies.values() for movie_id in movies))

        # Calculate the total number of hash functions required (bands × rows per band)
        self.num_hashes = bands * rows_per_band

        # Generate a set of hash functions for the MinHash signature matrix
        self.hash_funcs = generate_hash(self.num_hashes, self.p)

        # Store the user data and LSH parameters
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.user_movies = user_movies

        # Create LSH buckets and map users to buckets
        # `buckets` maps each bucket ID to the list of user IDs in that bucket
        # `user_buckets` maps each user ID to the list of bucket IDs they belong to
        self.buckets, self.user_buckets = get_buckets(self.user_movies, self.hash_funcs, self.bands)

    def top_k(self, user_id, k):
        """
        Find the top k users similar to a given user based on Jaccard similarity.

        Args:
        - user_id (int): The ID of the user to find similarities for.
        - k (int): The number of top similar users to return.

        Returns:
        - list: A list of the top k most similar user IDs,
                or None if fewer than k users are found.
        """
        similar_users = set()

        # Collect all users that share any bucket with the given user
        for bucket_id in self.user_buckets[user_id]:
            similar_users.update(self.buckets[bucket_id])

        # Remove the user itself from the set of similar users
        similar_users.discard(user_id)

        # If there are fewer than k similar users, return None
        if len(similar_users) < k:
            return None

        # Use a heap to maintain the top k most similar users based on Jaccard similarity
        top_k_similar = []

        for other_user in similar_users:
            # Compute Jaccard similarity between the given user and the other user
            similarity = jaccard_similarity(self.user_movies[user_id], self.user_movies[other_user])
            
            # Push the similarity and the user to the heap (negating similarity for max-heap behavior)
            heapq.heappush(top_k_similar, (similarity, other_user))

            # If there are more than k users in the heap, remove the least similar one
            if len(top_k_similar) > k:
                heapq.heappop(top_k_similar)

        # Extract the top k most similar users
        top_k_users = [user for _, user in top_k_similar]
        
        return top_k_users
    
    def collision_stats(self, num_random_users=1000, top_k=5):
        """
        Compute and display detailed statistics about user collisions in an LSH setup.

        This method analyzes a random subset of users to compute the following:
        - Percentage of users with no collisions.
        - Average number of collisions across sampled users.
        - Users with the most collisions (top k), including expected collisions and Jaccard similarities.
        - Users with the least collisions (top k), including expected collisions and Jaccard similarities.

        Parameters:
            num_random_users (int): Number of random users to sample for analysis. Default is 1000.
            top_k (int): Number of top users (most and least collisions) to display. Default is 5.

        Prints:
            - A summary of overall statistics (percentage of users with no collisions, average collisions).
            - A tabular report of the top k users with the most and least collisions, including:
            - Number of collisions.
            - Expected number of collisions based on Jaccard similarity.
            - Maximum Jaccard similarity across all users and collided users.
        """
        print_collision_stats(self.user_buckets, self.buckets, self.user_movies, self.threshold, num_random_users, top_k)



class RecommendationSystem:

    def __init__(self, lsh_classes, user_ratings, movie_titles):
        """
        Initialize the recommendation system with a list of LSH instances.

        Args:
            lsh_classes (list): A list of LSH class instances.
        """
        self.lsh_classes = lsh_classes
        self.user_movies = lsh_classes[0].user_movies
        self.user_ratings = user_ratings
        self.movie_titles = movie_titles

    def top_k(self, user_id, k=2):
        """
        Find the top k most similar users for the given user ID.

        The method attempts to find the top k most similar users using the `top_k` method 
        of each LSH instance in the list. If none of the LSH instances return valid results, 
        it falls back to a greedy method based on Jaccard similarity over all users.

        Args:
            user_id (int): The ID of the user for whom recommendations are sought.
            k (int): The number of top similar users to return.

        Returns:
            list: An ordered list of the top k most similar user IDs, or an empty list if no similar users are found.
        """
        # Try using LSH methods to find the top k most similar users
        for lsh in self.lsh_classes:
            similar_users = lsh.top_k(user_id, k=k)
            if similar_users is not None:
                return similar_users

        # Fallback: Use the greedy_top_k function for all users
        similar_users = greedy_top_k(user_id, self.user_movies, k=k)
        return similar_users
    
    def compare_greedy(self, num_random_users=10, k=2):
        """
        Compare the time complexity of the LSH-based `top_k` method with the purely greedy method.

        Args:
            num_random_users (int): Number of random users to sample for the comparison.
            k (int): Number of top similar users to find.

        Prints:
            - Average time taken by the LSH-based `top_k` method.
            - Average time taken by the purely greedy approach.
        """
        # Select random users to test
        all_users = list(self.lsh_classes[0].user_movies.keys())  # Assume all LSH instances share the same user_movies
        random_users = random.sample(all_users, min(num_random_users, len(all_users)))

        # Measure LSH-based `top_k` method time
        print("\nMeasuring LSH-based `top_k` performance...")
        lsh_times = []
        for user_id in tqdm(random_users, desc="LSH Top-k"):
            start_time = time.time()
            _ = self.top_k(user_id, k=k)
            lsh_time = time.time() - start_time
            lsh_times.append(lsh_time)

        # Measure greedy `top_k` time
        print("\nMeasuring purely greedy `top_k` performance...")
        user_movies = self.lsh_classes[0].user_movies  # Assume all LSH instances share the same user_movies
        greedy_times = []
        for user_id in tqdm(random_users, desc="Greedy Top-k"):
            start_time = time.time()
            _ = greedy_top_k(user_id, user_movies, k=k)
            greedy_time = time.time() - start_time
            greedy_times.append(greedy_time)

        # Compute and print average time for LSH and Greedy
        avg_lsh_time = sum(lsh_times) / len(lsh_times)
        avg_greedy_time = sum(greedy_times) / len(greedy_times)
        print(f"\nAverage time for LSH-based `top_k`: {avg_lsh_time:.3f} seconds")
        print(f"Average time for purely greedy `top_k`: {avg_greedy_time:.3f} seconds")

    def recommend(self, userId, max_films=5, show=False):
        similar_users = self.top_k(userId)
        user_seen_movies = set(self.user_movies[userId])

        # Get the movies seen by the two similar users, except the ones the query has already seen
        similar_user_1, similar_user_2 = similar_users        
        similar_user_1_movies = set(self.user_movies[similar_user_1]).difference(user_seen_movies)
        similar_user_2_movies = set(self.user_movies[similar_user_2]).difference(user_seen_movies)

        # Calculate intersection (movies both users have seen)
        common_movies = similar_user_1_movies.intersection(similar_user_2_movies)
        
        # Initialize list for recommended movies
        recommended_movies = []

        # If there are common movies, sort by average rating and select top max_films
        if common_movies:
            common_movies_data = []
            
            for movie_id in common_movies:
                avg_rating = (self.user_ratings[similar_user_1].get(movie_id, 0) +
                            self.user_ratings[similar_user_2].get(movie_id, 0)) / 2
                common_movies_data.append((movie_id, self.user_ratings[similar_user_1].get(movie_id, 0),
                                        self.user_ratings[similar_user_2].get(movie_id, 0), avg_rating))

            # Sort by average rating
            common_movies_data.sort(key=lambda x: x[3], reverse=True)

            # Select the top `max_films` movies
            recommended_movies = [movie[0] for movie in common_movies_data[:max_films]]

            # Print a tabular view of the intersection if show is True
            if show:
                print("-- Common movies (Intersection) - Sorted by Avg Rating \n")
                print(f"{'Movie ID':<10} {'Title':<30} {'User 1 Rating':<15} {'User 2 Rating':<15} {'Avg Rating':<15}")
                print("-" * 80)    
                for movie in common_movies_data[:max_films]:
                    title = self.movie_titles.get(movie[0], 'Unknown Title')
                    truncated_title = title[:25] + ('...' if len(title) > 25 else '')  # Truncate titles to 25 chars        
                    print(f"{movie[0]:<10} {truncated_title:<30} {movie[1]:<15} {movie[2]:<15} {movie[3]:<15.2f}")    
                if len(common_movies_data) >= max_films:
                    print(f"\nSufficient movies found in the intersection: {max_films} recommendations.")
                    return recommended_movies
                else:
                    print(f"\nLess than {max_films} found in intersection. Checking symmetric difference...")

        else:
            print("No movie in the intersection. Checking symmetric difference...")

        # If we don't have enough recommendations, add from the symmetric difference

        # Get symmetric difference (movies only seen by one of the two users)
        movies_diff = similar_user_1_movies.symmetric_difference(similar_user_2_movies)

        # If there are movies in the symmetric difference, sort them by individual user ratings
        if movies_diff:
            diff_movies_data = []
            for movie_id in movies_diff:
                rating_user_1 = self.user_ratings[similar_user_1].get(movie_id, 0)
                rating_user_2 = self.user_ratings[similar_user_2].get(movie_id, 0)
                rating = max(rating_user_1, rating_user_2)  
                diff_movies_data.append((movie_id, rating_user_1, rating_user_2, rating))

            # Sort by highest individual rating
            diff_movies_data.sort(key=lambda x: x[3], reverse=True)

            # Print a tabular view of the symmetric difference if show is True
            if show:
                print("\n-- Movies from Symmetric Difference - Sorted by Rating \n")
                print(f"{'Movie ID':<10} {'Title':<30} {'User 1 Rating':<15} {'User 2 Rating':<15}")
                print("-" * 80)    
                for movie in diff_movies_data[:max_films - len(recommended_movies)]:
                    title = self.movie_titles.get(movie[0], 'Unknown Title')
                    truncated_title = title[:25] + ('...' if len(title) > 25 else '')        
                    print(f"{movie[0]:<10} {truncated_title:<30} {movie[1]:<15} {movie[2]:<15}")       

            # Add remaining movies to the recommendations
            recommended_movies.extend([movie[0] for movie in diff_movies_data[:max_films - len(recommended_movies)]])
        else: 
            print(f"\nSymmetric difference is empty")

        # Return the recommended movies
        return recommended_movies



def get_buckets(user_movies, hash_funcs, bands):
    """
    Perform Locality-Sensitive Hashing (LSH) using MinHash signatures to group similar users into buckets.

    This function computes MinHash signatures for users' sets of movies, applies LSH banding, and distributes users into buckets. 
    It also stores the bucket IDs for each user to facilitate further similarity analysis.
    The function precomputes the MinHash signatures for all movies to avoid redundant calculations and uses vectorized 
    operations with NumPy to speed up the signature generation and banding process.

    Parameters:
        user_movies (dict): A dictionary where keys are user IDs and values are sets of movies watched by the user.
        hash_funcs (list): A list of hash functions to compute the MinHash signatures.
        bands (int): The number of bands to use in LSH banding.

    Returns:
        tuple:
            - buckets (defaultdict): A mapping from bucket IDs to lists of users in those buckets.
            - user_buckets (defaultdict): A mapping from user IDs to the list of bucket IDs where the user is placed.
    """

    buckets = defaultdict(list)
    user_buckets = defaultdict(list)  # To store the list of buckets for each user

    # Precompute number of rows per band
    r = len(hash_funcs) // bands

    # Step 1: Precompute the hashes for all movies
    all_movies = set(movie for movies in user_movies.values() for movie in movies)  # Collect all unique movies
    movie_hashes = {} 
    for movie in tqdm(all_movies, total=len(all_movies), desc="Precomputing Movie Hashes"):
        movie_hashes[movie] = [hash_func(movie) for hash_func in hash_funcs]

    # Convert movie hashes to NumPy arrays for faster operations
    movie_hashes_np = {movie: np.array(hashes) for movie, hashes in movie_hashes.items()}

    # Step 2: Generate MinHash signatures and apply LSH banding
    for user, movies in tqdm(user_movies.items(), total=len(user_movies), desc="Hashing Users into Buckets"):
        # Convert movies to NumPy array
        movies_np = np.array(list(movies))

        # Get the hash values for all movies
        movie_hash_values = np.array([movie_hashes_np[movie] for movie in movies_np])

        # Calculate the MinHash signature for all hash functions at once (min across rows for each column)
        sig = movie_hash_values.min(axis=0)  # This is equivalent to the 'min' over each column

        #Apply LSH banding
        for i in range(bands):
            band = tuple(sig[i * r:(i + 1) * r])
            bucket_id = hash((i, band))  # Include band index in the hash
            buckets[bucket_id].append(user)
            user_buckets[user].append(bucket_id)  # Store the bucket ID for the user

    return buckets, user_buckets



def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.

    The Jaccard similarity is defined as the size of the intersection of two sets 
    divided by the size of their union.

    Args:
    - set1 (set): The first set of elements.
    - set2 (set): The second set of elements.

    Returns:
    - float: The Jaccard similarity between set1 and set2, a value between 0 and 1.
            - 1 means the sets are identical.
            - 0 means the sets are disjoint.
    """
    return len(set1 & set2) / len(set1 | set2)



def calculate_jaccard_and_collisions(user_id, similar_users, user_buckets, user_movies, threshold=0.5):
    """
    Calculate Jaccard similarity-related statistics for a specific user.

    This function computes the following metrics:
    - The number of users with Jaccard similarity above a given threshold.
    - The maximum Jaccard similarity across all users.
    - The maximum Jaccard similarity among users that share at least one bucket with the target user.

    Parameters:
        user_id (int): The ID of the target user.
        similar_users (set): A set of user IDs sharing at least one bucket with the target user.
        user_buckets (dict): A dictionary mapping user IDs to their associated bucket IDs.
        user_movies (dict): A dictionary mapping user IDs to sets of movie IDs watched by each user.
        threshold (float): The Jaccard similarity threshold to count expected collisions. Default is 0.5.

    Returns:
        tuple:
            - expected_collisions (int): Number of users with Jaccard similarity >= threshold.
            - max_similarity_all (float): Maximum Jaccard similarity across all users.
            - max_similarity_collided (float): Maximum Jaccard similarity among collided users.
    """
    max_similarity_all = 0.0
    max_similarity_collided = 0.0
    expected_collisions = 0

    # Iterate over all users in the dataset
    for other_user_id in user_buckets.keys():
        if other_user_id != user_id:
            # Compute the Jaccard similarity between the target user and the current user
            similarity = jaccard_similarity(user_movies[user_id], user_movies[other_user_id])
            
            # Update the maximum similarity across all users
            max_similarity_all = max(max_similarity_all, similarity)
            
            # Count users with similarity above the threshold
            if similarity >= threshold:
                expected_collisions += 1
            
            # Update the maximum similarity among users that share buckets with the target user
            if other_user_id in similar_users:
                max_similarity_collided = max(max_similarity_collided, similarity)

    return expected_collisions, max_similarity_all, max_similarity_collided



def print_collision_stats(user_buckets, buckets, user_movies, threshold, num_random_users, top_k):
    """
    Analyze and display statistics on user collisions in an LSH (Locality Sensitive Hashing) setup.

    This function computes collision statistics among users based on shared buckets. 
    It also calculates expected collisions and Jaccard similarity metrics for users 
    with the highest and lowest number of collisions.

    Parameters:
        user_buckets (defaultdict): 
            Maps user IDs to lists of bucket IDs they belong to.
        buckets (defaultdict): 
            Maps bucket IDs to lists of user IDs contained within each bucket.
        user_movies (dict): 
            Maps user IDs to sets of movie IDs they have watched.
        threshold (float): 
            The Jaccard similarity threshold to determine if two users are considered to have collided.
        num_random_users (int): 
            The number of users to randomly sample for analysis.
        top_k (int): 
            The number of users with the most and least collisions to display.

    Prints:
        - The percentage of sampled users with no collisions.
        - The average number of collisions per user.
        - A tabular list of the top `top_k` users with the most collisions, including:
            * Total collisions
            * Expected collisions (based on Jaccard similarity threshold)
            * Maximum Jaccard similarity across all users
            * Maximum Jaccard similarity among collided users
        - A tabular list of the top `top_k` users with the least collisions with the same metrics.
    """
    # Randomly sample a subset of users for analysis
    random.seed(42)
    random_users = random.sample(list(user_buckets.keys()), num_random_users)

    # Track the number of users sharing buckets with each sampled user
    shared_user_counts = []

    # Compute shared users for each sampled user
    for user_id in tqdm(random_users, desc="Sampling Users for Stats", total=num_random_users):
        shared_users = set()

        # Collect all users sharing buckets with the current user
        for bucket_id in user_buckets[user_id]:
            bucket_users = buckets[bucket_id]
            for other_user_id in bucket_users:
                if other_user_id != user_id:
                    shared_users.add(other_user_id)

        # Append user statistics
        shared_user_counts.append((user_id, len(shared_users), shared_users))

    # Calculate overall statistics on collisions
    no_shared_users_count = sum(1 for _, count, _ in shared_user_counts if count == 0)
    percentage_no_shared = (no_shared_users_count / num_random_users) * 100
    avg_shared = sum(count for _, count, _ in shared_user_counts) / len(shared_user_counts)

    # Sort users by their collision counts
    sorted_shared_user_counts = sorted(shared_user_counts, key=lambda x: x[1])

    # Identify users with the most and least collisions
    most_collisions_users = sorted_shared_user_counts[-top_k:]
    least_collisions_users = sorted_shared_user_counts[:top_k]

    # Display overall collision statistics
    print(f"\tPercentage of users with no collisions: {percentage_no_shared:.1f}%")
    print(f"\tAverage collisions: {avg_shared:.2f}")

    # Set Pandas display format for floats
    pd.options.display.float_format = '{:.3f}'.format

    # Compute and display details for users with the most collisions
    print('\n')
    df_most_collisions = []
    for user_id, collisions, similar_users in tqdm(most_collisions_users, desc='Analyzing stats for users with most collisions'):
        expected_collisions_val, max_similarity_all, max_similarity_collided = calculate_jaccard_and_collisions(user_id, similar_users, user_buckets, user_movies)
        df_most_collisions.append([user_id, collisions, expected_collisions_val, max_similarity_all, max_similarity_collided])

    df_most_collisions_df = pd.DataFrame(df_most_collisions, columns=["User ID", "Collision", "Exp_Coll", "Max_Sim_All", "Max_Sim_Coll"])
    print(f"\t{top_k} users with most collisions:\n")
    print(df_most_collisions_df.to_string(index=False))

    # Compute and display details for users with the least collisions
    print('\n')
    df_least_collisions = []
    for user_id, collisions, similar_users in tqdm(least_collisions_users, desc='Analyzing stats for users with least collisions'):
        expected_collisions_val, max_similarity_all, max_similarity_collided = calculate_jaccard_and_collisions(user_id, similar_users, user_buckets, user_movies)
        df_least_collisions.append([user_id, collisions, expected_collisions_val, max_similarity_all, max_similarity_collided])

    df_least_collisions_df = pd.DataFrame(df_least_collisions, columns=["User ID", "Collision", "Exp_Coll", "Max_Sim_All", "Max_Sim_Coll"])
    print(f"\t{top_k} users with least collisions: \n")
    print(df_least_collisions_df.to_string(index=False))



def greedy_top_k(user_id, user_movies, k=2):
    """
    Find the top k most similar users to the given user using Jaccard similarity.

    Args:
        user_id (int): The ID of the user for whom recommendations are sought.
        user_movies (dict): A dictionary mapping user IDs to sets of movie IDs they have rated.
        k (int): The number of top similar users to find (default is 2).

    Returns:
        list: A list of the top k most similar user IDs, or an empty list if no users are found.
    """
    user_set = user_movies.get(user_id, set())
    top_k_heap = []

    for other_user, other_movies in user_movies.items():
        if other_user == user_id:
            continue
        # Compute Jaccard similarity
        similarity = jaccard_similarity(user_set, other_movies)
        # Use the heap to maintain the top k most similar users
        heapq.heappush(top_k_heap, (similarity, other_user))
        if len(top_k_heap) > k:
            heapq.heappop(top_k_heap)

    # Extract the top k users from the heap (if they exist)
    if top_k_heap:
        top_k_users = [heapq.heappop(top_k_heap)[1] for _ in range(len(top_k_heap))]
        top_k_users.reverse()  # Reverse since heap pops the smallest first
        return top_k_users
    return []