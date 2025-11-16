import sys
sys.path.append('./aima-python')  # Ensure correct path to aima-python

# Importing necessary modules from the AIMA library
from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, rejection_sampling, likelihood_weighting, prior_sample

# Task 1: Define a Bayesian Network and Query it

# Define a custom Bayesian network
# This example uses a simple disease diagnosis model.
# Variables:
#   - "Disease": Whether a person has a disease
#   - "Test": Whether a test result is positive, which depends on having the disease

# Define the BayesNet structure
disease_network = BayesNet([
    ('Disease', '', 0.01),  # Prior probability of having the disease is 1%
    ('Test', 'Disease', {True: 0.95, False: 0.1})  # Sensitivity and specificity of the test
])

# Function to demonstrate querying the BayesNet
def query_disease_network():
    print("Exact Inference (Enumeration) - P(Disease | Test=True):")
    print(enumeration_ask('Disease', {'Test': True}, disease_network).show_approx())

    print("\nExact Inference (Variable Elimination) - P(Disease | Test=True):")
    print(elimination_ask('Disease', {'Test': True}, disease_network).show_approx())

    print("\nApproximate Inference (Rejection Sampling) - P(Disease | Test=True):")
    print(rejection_sampling('Disease', {'Test': True}, disease_network, N=1000).show_approx())

    print("\nApproximate Inference (Likelihood Weighting) - P(Disease | Test=True):")
    print(likelihood_weighting('Disease', {'Test': True}, disease_network, N=1000).show_approx())

    print("\nApproximate Inference (Gibbs Sampling) - P(Disease | Test=True):")
    print(gibbs_ask('Disease', {'Test': True}, disease_network, N=1000).show_approx())

# Run the query function
query_disease_network()

# Task 2: Create an Alternate Example of a BayesNet and Query

# Define a different Bayesian network for weather and activity choice
# Variables:
#   - "Rain": Probability it rains on a given day
#   - "Sprinkler": Probability the sprinkler is on, depends on rain
#   - "Grass Wet": Probability the grass is wet, depends on rain and sprinkler

weather_network = BayesNet([
    ('Rain', '', 0.2),  # 20% chance of rain
    ('Sprinkler', 'Rain', {True: 0.01, False: 0.4}),  # Sprinkler is more likely if it didn't rain
    ('Grass Wet', 'Rain Sprinkler', {
        (True, True): 0.99,  # Both rain and sprinkler on, grass is wet
        (True, False): 0.9,  # Rain but no sprinkler, grass still likely wet
        (False, True): 0.8,  # Sprinkler but no rain, grass is likely wet
        (False, False): 0.0  # No rain and no sprinkler, grass is dry
    })
])

# Function to demonstrate querying the weather network
def query_weather_network():
    print("Exact Inference (Enumeration) - P(Rain | Grass Wet=True):")
    print(enumeration_ask('Rain', {'Grass Wet': True}, weather_network).show_approx())

    print("\nExact Inference (Variable Elimination) - P(Rain | Grass Wet=True):")
    print(elimination_ask('Rain', {'Grass Wet': True}, weather_network).show_approx())

    print("\nApproximate Inference (Rejection Sampling) - P(Rain | Grass Wet=True):")
    print(rejection_sampling('Rain', {'Grass Wet': True}, weather_network, N=1000).show_approx())

    print("\nApproximate Inference (Likelihood Weighting) - P(Rain | Grass Wet=True):")
    print(likelihood_weighting('Rain', {'Grass Wet': True}, weather_network, N=1000).show_approx())

    print("\nApproximate Inference (Gibbs Sampling) - P(Rain | Grass Wet=True):")
    print(gibbs_ask('Rain', {'Grass Wet': True}, weather_network, N=1000).show_approx())

# Run the query function
query_weather_network()
