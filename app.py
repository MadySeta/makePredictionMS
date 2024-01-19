from flask import Flask, jsonify, request
import pickle
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('./finalModel/best_rf_model.pkl', 'rb'))
scaler = pickle.load(open('./finalModel/scaler', 'rb'))

@app.route('/predictionUser', methods=['GET'])
def prediction_user():
    user_data = request.get_json()
    df = pd.DataFrame.from_dict([user_data])
    scaled_data = scaler.transform(df)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)

    result = {
        'prediction': predictions[0],
        'trust_bot': round(probabilities[0][0], 4),
        'trust_human': round(probabilities[0][1], 4)
    }

    return jsonify(result)


@app.route('/predictionUserFollowers', methods=['GET'])
def prediction_user_followers():
    followers_data = request.get_json()
    all_predictions = []
    all_human_probabilities = []
    all_bot_probabilities = []

    for follower_data in followers_data:
        predictions, probabilities = prediction_user({'user_data': follower_data})
        all_predictions.append(predictions[0])
        if predictions[0] == 'human':
            all_human_probabilities.append(probabilities[0][1])
        else:
            all_bot_probabilities.append(probabilities[0][0])

    result = {
        'all_predictions': all_predictions,
        'all_human_probabilities': all_human_probabilities,
        'all_bot_probabilities': all_bot_probabilities
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
