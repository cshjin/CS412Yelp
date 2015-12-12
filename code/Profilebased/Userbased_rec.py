from crab import datasets
from crab.models import MatrixPreferenceDataModel
from crab.metrics import pearson_correlation
from crab.similarities import UserSimilarity
from crab.recommenders.knn import UserBasedRecommender
from crab.metrics.classes import CfEvaluator

# load rating matrix
reviews = datasets.load_reviews()

# Build the model
model = MatrixPreferenceDataModel(reviews.data)

# Build the similarity
similarity = UserSimilarity(model, pearson_correlation, 100)

# Build the User based recommender
recommender = UserBasedRecommender(model, similarity, with_preference=True)

# Recommend items for the user 3
print recommender.recommend(3)

# evaluation
evaluator = CfEvaluator()
rmse = evaluator.evaluate(recommender, metric='rmse', permutation=False,  sampling_ratings=0.8)
print rmse
# rmse = evaluator.evaluate_on_split(recommender, metric='rmse', permutation=False,)
# print rmse