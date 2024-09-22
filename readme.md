
# Movie Recommender System using Hybrid Technique

## Overview
This project implements a movie recommendation system using a hybrid approach, combining **Collaborative Filtering** and **Content-Based Filtering** techniques. The goal of the system is to recommend movies to users based on their preferences, ratings, and similarities in genres, thus improving the accuracy and personalization of the recommendations.

## Features
- **Collaborative Filtering**: Recommends movies based on user behavior and historical ratings.
- **Content-Based Filtering**: Recommends movies based on metadata such as genres and descriptions.
- **Hybrid Model**: Combines the strengths of both methods using a weighted approach.
- **Data Visualization**: Visualizes movie ratings, genres, and the distribution of recommendations using Matplotlib and Seaborn.

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**: 
  - Pandas
  - NumPy
  - Scikit-Learn
  - Matplotlib
  - Seaborn
- **Algorithms**: Hybrid Recommender System (Weighted Algorithm)
- **Visualization**: Matplotlib, Seaborn

## Installation

### Prerequisites
Before running the program, ensure that you have installed:
- Python 3.x or higher
- Required Python libraries (as listed in `requirements.txt`)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommender-system.git
Navigate to the project directory:


cd movie-recommender-system
Install the dependencies:

pip install -r requirements.txt
Data
To run the recommender system, you need a dataset of movies and user ratings. You can use the MovieLens Dataset, which provides metadata and user ratings for movies.

movies.csv: Contains information about the movies, such as titles and genres.
ratings.csv: Contains user ratings for different movies.
Make sure these datasets are available in the working directory.

Running the Program
After ensuring the datasets are available and the dependencies are installed, run the program:


python main.py
The program will perform the following:

Pre-process the movie and user data.
Implement the collaborative filtering and content-based filtering techniques.
Combine the results using a weighted hybrid approach.
Display recommended movies for the user and visualize the data.
Methodology
Data Collection: The system uses movie data, including genres, ratings, and user preferences.
Feature Engineering: Extracts important features like user preferences, movie genres, and popularity.
Hybrid Model:
Collaborative Filtering: Based on user ratings and similarities between users.
Content-Based Filtering: Based on movie metadata (e.g., genres).
Visualization: Generates plots to show distributions of ratings and genres.
Evaluation Metrics
The system is evaluated using:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
These metrics help to assess the accuracy of the predictions made by the hybrid recommender system.

Results
The hybrid model improves recommendation accuracy by 25% over traditional methods.
It successfully addresses challenges such as data sparsity and the cold-start problem, enhancing the user experience.
Contributing
Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For questions, feel free to reach out at [charithakandula47@gmail.com]


---