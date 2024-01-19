import pickle
import pandas as pd
# Load the model from the pickle file

path_fo_file = "/home/asmun/Desktop/INSA/5A/ProjetIntegrator/appFlask/finalModel"

loaded_model  = pickle.load(open(path_fo_file + '/best_rf_model.pkl','rb'))
scaler = pickle.load(open(path_fo_file + '/scaler','rb'))

feature_names = ['default_profile', 'default_profile_image', 'favourites_count', 
                 'followers_count', 'friends_count', 'geo_enabled',
                 'statuses_count', 'verified', 'average_tweets_per_day', 
                 'account_age_days']

# Replace this with your actual test data
test_data = pd.DataFrame([[
            False,   # default_profile
            False,   # default_profile_image
            100,     # favourites_count
            150,     # followers_count
            75,      # friends_count
            True,    # geo_enabled
            500,     # statuses_count
            False,   # verified
            5,       # average_tweets_per_day
            365      # account_age_days
        ]], columns=feature_names)

test_data = scaler.transform(test_data)

predictions = loaded_model.predict(test_data)
probabilities = loaded_model.predict_proba(test_data)

# Assuming a binary classification (0 or 1), adjust if more classes
# This extracts the probability of the predicted class

print("Predictions:", predictions)
print("Trust Percentages:", probabilities)  


