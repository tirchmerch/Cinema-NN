# Cinema-NN
CS4343 Final Project - Cinema Neural Network

This Project demonstrates the difference between models trained exclusively on user, critic, and user + critic movie reviews.
There are 6 total models:
- CNN (User Reviews)
- CNN (Critic Reviews)
- CNN (Both Reviews)
- RNN (User Reviews)
- RNN (Critic Reviews)
- RNN (Both Reviews)

The dataset utilized was: https://www.kaggle.com/datasets/davutb/metacritic-movies?select=movies.csv

## rnn_cnn.ipynb

This file provides all features and imports to train both the CNN and RNN models on 10k samples of user data and 10k samples of critic data, for a total 0f 20k samples for the model trained on both. 

## rnn_model.ipynb

This model is exclusively the RNN model and also includes hyperparameter tuning to find the optimal hyperparameters for the user, critic, and combined models.
