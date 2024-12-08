# ADM-HW4

The present repository contains the solutions to the [4th ADM assignment](https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_4) for the year 2024/2025.

#### Collaborators (Group 12):
- Leonardo Rocci ([cycileo](https://github.com/cycileo))
- Emre Yesil ([1emreyesil](https://github.com/1emreyesil))   
- Antonio Pagnotta ([AntonioPagnotta](https://github.com/AntonioPagnotta))  
- Zuzana Miciakova ([Bioinf-Suzy](https://github.com/Bioinf-Suzy))    

## Files Overview
- `main.ipynb`: This is the main notebook containing the solutions to the homework, along with a command to install the required packages.
- `requirements.txt`: This file lists the necessary packages and libraries needed to run the notebook.
- `lsh.py`: This file contains the function implementations for the first part of the homework (LSH-based recommendation system).
- `kmeans_animation.gif`: A gif that visualizes the clustering process for the third section of the homework (bonus question).
- The notebook automatically downloads the dataset and saves it inside a folder named `Data` when run.

## Homework Abstract: ADM-HW4 - Movie Recommendation System

### Section 1: Recommendation System with LSH
#### Data Preparation:
- Download and explore the MovieLens dataset.

#### Minhash Signatures:
- Implement a MinHash function to create signature vectors for each user based on their rated movies.

#### Locality-Sensitive Hashing (LSH):
- Use Locality-Sensitive Hashing (LSH) to cluster similar users.
- Implement bucket creation and find similar users for recommendations.
- If common movies exist between users, recommend based on average ratings; otherwise, recommend top-rated movies from similar users.

### Section 2: Grouping Movies Together
#### Feature Engineering:
- Create features to represent movies, such as movie genres and average ratings.

#### Feature Selection & Normalization:
- Decide whether to normalize features or reduce dimensionality using techniques like PCA (Principal Component Analysis).

#### Clustering:
- Implement K-means and K-means++.
- Evaluate clustering quality using various metrics and compare results.

#### Best Algorithm:
- Assess the optimal clustering algorithm based on evaluation metrics.

### Section 3: Bonus Question
- Track cluster progression across iterations in a 2D plot, using the most effective features for visualization.

### Section 4: Algorithmic Question
- Develop a strategy for Arya to guarantee a win in a game where players pick numbers from the ends of a sequence.
- Implement the optimal strategy, analyze its efficiency, and optimize it further with recommendations from Large Language Models (LLMs).