BiteMe Model Card
Taste Inference & Food Understanding System
Model Overview

The BiteMe AI system infers a user's food preferences from visual and interaction signals in order to build a personalized Flavor Fingerprint, which is used to recommend restaurants and compatible dining partners.

The system combines multiple machine learning components:

Vision Encoder

A CLIP-based vision transformer converts food images into embeddings in a shared vision-language space.

Embedding-Based Retrieval

Image embeddings are compared against an indexed library of dish embeddings using nearest-neighbor similarity. This produces candidate dish predictions.

Preference Inference Model

Predicted dishes and explicit user signals are translated into a structured taste representation composed of cuisine preferences, dish affinities, and latent flavor dimensions.

Recommendation Layer

The taste profile feeds a compatibility scoring model used to rank restaurants and other users.

The system therefore combines:

Vision-language representation learning

Embedding retrieval

Preference learning

Recommendation scoring

This architecture is best described as an embedding-based food understanding system with preference inference and recommendation ranking.

Intended Use

The AI model supports personalized food discovery and social dining recommendations.

Its role in the product is to:

Infer taste preferences from food images and user interactions

Construct a dynamic taste profile (Flavor Fingerprint)

Provide signals for recommendation systems that match users with:

restaurants aligned with their taste

other users with compatible preferences

The system is intended for consumers exploring restaurants or meeting people with similar food interests.

The model is designed for preference inference and recommendation, not:

nutritional analysis

dietary advice

medical or health decisions

Data
Training Data

The visual understanding component relies on:

publicly available food image datasets

dish category datasets containing labeled food images

CLIP pretraining data for vision-language representations

These datasets contain thousands of food images spanning common cuisines and presentation styles.

The model leverages pretrained CLIP embeddings, which were originally trained on large-scale image-text pairs.

Inference Data

At runtime the model processes three types of signals:

Image Inputs

User-uploaded food photos.

Explicit Feedback

Users selecting recommended dishes.

Behavioral Signals

Interaction history including restaurant selections and preference signals.

These inputs are transformed into embeddings and preference signals used to update the taste profile.

Known Data Limitations

Food datasets introduce several limitations:

Cuisine Representation Imbalance

Some cuisines appear more frequently than others.

Presentation Bias

Images often represent professionally plated restaurant food rather than everyday meals.

Closed-Set Dish Space

Because the model retrieves from a fixed dish vocabulary, non-food images may initially map to food classes.

To mitigate this, a food vs non-food rejection gate was implemented.

Evaluation

Evaluation focuses on both prediction accuracy and product usefulness.

Dish Prediction

The visual retrieval model is evaluated using:

Top-1 Accuracy — whether the highest scoring dish prediction is correct

Top-3 Accuracy — whether the correct dish appears among the top three candidates

Top-3 accuracy is particularly relevant because the system uses candidate sets rather than relying solely on a single label.

Taste Profile Responsiveness

Because the model's ultimate goal is recommendation quality, we also evaluate:

profile responsiveness to new food signals

stability of preference updates

consistency of recommendations with user expectations

A “good” result means:

obvious foods produce intuitive preference updates

repeated signals strengthen the correct taste dimensions

restaurant recommendations align with predicted tastes.

Performance & Limitations
Strengths

The system performs well when:

images contain clearly recognizable dishes

foods belong to common categories such as pizza, ramen, or desserts

users provide multiple signals over time

In these cases the model reliably produces consistent taste profiles and intuitive restaurant recommendations.

Limitations
Non-Food Images

Because dish retrieval occurs within a food embedding space, non-food images may sometimes resemble food classes.

A rejection mechanism now detects and filters non-food images before updating user profiles.

Ambiguous Foods

Visually similar dishes (for example noodle dishes across cuisines) may produce uncertain predictions.

Cold Start Users

Users with limited interaction history initially receive more generic recommendations until the system collects enough signals.

Dataset Bias

Food datasets emphasize visually distinctive dishes and restaurant-quality imagery.

Home cooking and culturally specific dishes may be underrepresented.

Improvement Path
Implemented Improvement

During development we observed that non-food images occasionally produced dish predictions due to the closed-set retrieval approach.

To address this we implemented a food-vs-non-food gate using CLIP similarity comparisons between food prompts and non-food prompts.

If an image scores higher against non-food prompts, the system returns not_food and prevents profile updates.

This significantly reduced incorrect taste-profile updates.

Future Improvements

Several improvements could increase system robustness:

Expanded Dish Dataset

Increasing coverage of global cuisines and dishes.

Hybrid Classification

Combining retrieval-based predictions with supervised dish classifiers.

User Feedback Loop

Allowing users to correct incorrect dish predictions.

Behavioral Learning

Training preference models on larger user interaction datasets to improve recommendation accuracy.

Summary

The BiteMe AI system integrates vision-language embeddings, retrieval-based food understanding, and preference inference to build a personalized taste profile. This profile powers restaurant and social dining recommendations. While the system performs well for common food categories and clear visual signals, limitations remain in dataset coverage, ambiguous dishes, and cold-start scenarios. Continued improvements focus on expanding training data, improving classification robustness, and incorporating richer user feedback.