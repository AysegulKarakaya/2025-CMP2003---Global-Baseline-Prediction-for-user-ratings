#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

// Data structures
struct Rating {
    int userID;
    int movieID;
    double rating;
};

// Function to compute cosine similarity between two items
double computeCosineSimilarity(const vector<double>& item1Ratings, const vector<double>& item2Ratings) {
    double dotProduct = 0.0, magnitudeItem1 = 0.0, magnitudeItem2 = 0.0;

    for (size_t i = 0; i < item1Ratings.size(); ++i) {
        dotProduct += item1Ratings[i] * item2Ratings[i];
        magnitudeItem1 += item1Ratings[i] * item1Ratings[i];
        magnitudeItem2 += item2Ratings[i] * item2Ratings[i];
    }

    if (magnitudeItem1 == 0.0 || magnitudeItem2 == 0.0) return 0.0; // Avoid division by zero
    return dotProduct / (sqrt(magnitudeItem1) * sqrt(magnitudeItem2));
}

// Function to predict the rating of a user for a specific movie
double predictRating(int userID, int movieID,
    const unordered_map<int, unordered_map<int, double>>& userRatings,
    const unordered_map<int, vector<double>>& movieRatingsMatrix) {

    if (movieRatingsMatrix.find(movieID) == movieRatingsMatrix.end()) return 0.0; // Movie not found

    vector<double> targetMovieRatings = movieRatingsMatrix.at(movieID);
    double weightedSum = 0.0, similaritySum = 0.0;

    for (const auto& movieEntry : movieRatingsMatrix) {
        int otherMovieID = movieEntry.first;
        const vector<double>& otherMovieRatings = movieEntry.second;

        if (movieID == otherMovieID) continue; // Skip the target movie

        // Calculate similarity between the target movie and the other movie
        double similarity = computeCosineSimilarity(targetMovieRatings, otherMovieRatings);

        // Check if the user rated the other movie
        if (userRatings.at(userID).find(otherMovieID) != userRatings.at(userID).end()) {
            weightedSum += similarity * userRatings.at(userID).at(otherMovieID);
            similaritySum += abs(similarity);
        }
    }

    return (similaritySum == 0.0) ? 0.0 : (weightedSum / similaritySum); // Avoid division by zero
}

// Load training data from a file
void loadTrainingData(const string& filename,
    unordered_map<int, unordered_map<int, double>>& userRatings,
    unordered_map<int, vector<double>>& movieRatingsMatrix) {
    ifstream file(filename);
    string line;

    // Map to track all user ratings for movies
    unordered_map<int, vector<double>> tempMovieRatings;

    // Track unique user IDs to ensure proper indexing
    unordered_map<int, int> userIndexMap;
    int userIndex = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        int userID, movieID;
        double rating;
        char delimiter;

        ss >> userID >> delimiter >> movieID >> delimiter >> rating;

        // Populate user ratings
        userRatings[userID][movieID] = rating;

        // Assign a unique index to each user ID
        if (userIndexMap.find(userID) == userIndexMap.end()) {
            userIndexMap[userID] = userIndex++;
        }

        // Dynamically adjust the size of tempMovieRatings
        int userMappedIndex = userIndexMap[userID];
        if (tempMovieRatings[movieID].size() <= userMappedIndex) {
            tempMovieRatings[movieID].resize(userMappedIndex + 1, 0.0);
        }
        tempMovieRatings[movieID][userMappedIndex] = rating;
    }

    // Build movie ratings matrix
    for (const auto& movieEntry : tempMovieRatings) {
        int movieID = movieEntry.first;
        const vector<double>& ratings = movieEntry.second;
        movieRatingsMatrix[movieID] = ratings;
    }
}

int main() {
    // File paths (update as needed)
    string trainingDataFile = "training_data.csv";
    string testDataFile = "test_data.csv";

    // Data structures
    unordered_map<int, unordered_map<int, double>> userRatings; // User -> (Movie -> Rating)
    unordered_map<int, vector<double>> movieRatingsMatrix; // Movie -> Ratings vector

    // Load training data
    loadTrainingData(trainingDataFile, userRatings, movieRatingsMatrix);

    // Predict ratings for test data
    ifstream testFile(testDataFile);
    ofstream outputFile("predicted_ratings.csv");
    string line;

    while (getline(testFile, line)) {
        stringstream ss(line);
        int userID, movieID;
        char delimiter;

        ss >> userID >> delimiter >> movieID;

        double predictedRating = predictRating(userID, movieID, userRatings, movieRatingsMatrix);
        outputFile << userID << "," << movieID << "," << predictedRating << endl;
    }

    cout << "Predictions completed. Check predicted_ratings.csv for results." << endl;
    return 0;
}
