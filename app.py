from flask import Flask, jsonify, request
import pickle
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('./finalModel/best_rf_model.pkl', 'rb'))
scaler = pickle.load(open('./finalModel/scaler', 'rb'))

def arrangeFeatures(df : pd.DataFrame):
    order = ['default_profile', 
            'default_profile_image',
            'favourites_count',
            'followers_count',
            'friends_count',
            'geo_enabled',
            'statuses_count',
            'verified',
            'average_tweets_per_day',
            'account_age_days']
    return df[order]


def prediction(user_data : dict):
    df = pd.DataFrame.from_dict([user_data])
    df_ordered = arrangeFeatures(df)
    scaled_data = scaler.transform(df_ordered)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)

    result = {
        'prediction': predictions[0],
        'trust_bot': round(probabilities[0][0], 4),
        'trust_human': round(probabilities[0][1], 4)
    }
    return result

@app.route('/predictionUser', methods=['GET'])
def prediction_user():
    user_data = request.get_json()
    result = prediction(user_data)

    return jsonify(result)


@app.route('/predictionUserFollowers', methods=['GET'])
def prediction_user_followers():
    followers_data = request.get_json()
    
    all_predictions = []
    all_human_probabilities = []
    all_bot_probabilities = []

    for follower_data in followers_data:
        result = prediction(follower_data)
        if result['prediction'] == 'human' and result['trust_human'] >= 0.7:
            all_predictions.append('human')
            all_human_probabilities.append(result['trust_human'])
        elif result['prediction'] == 'bot' and result['trust_bot'] >= 0.6:
            print('---- HERE ---- ')
            all_predictions.append('bot')
            all_bot_probabilities.append(result['trust_bot'])
        else:
            all_predictions.append('undefined')

    result = {
        'all_predictions': all_predictions,
        'all_human_probabilities': all_human_probabilities,
        'all_bot_probabilities': all_bot_probabilities
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
