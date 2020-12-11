import json
import numpy as np
import boto3
from flask import Flask, request
from sklearn.gaussian_process import GaussianProcessRegressor

app = Flask(__name__)

client = boto3.client('dynamodb', region_name='us-east-2')


@app.route("/")
def index():
    return "Smart Plant API"


@app.route("/health")
def health():
    return 200

@app.route("/sensor")
def get_sensor():
    row = str(request.args.get('row'))
    pos = str(request.args.get('pos'))
    resp = client.get_item(
        TableName='SoilSensor',
        Key={
            'Row': {'S': row},
            'PositionInRow': {'S': pos}
        }
    )
    soil_data = np.array([float(i['N']) for i in resp['Item']['payload']['M']['moistureSensor']['L']]).reshape(-1, 1)
    light_data = np.array([float(i['N']) for i in resp['Item']['payload']['M']['lightSensor']['L']]).reshape(-1, 1)
    x = np.arange(len(soil_data)).reshape(-1, 1)
    predictive_range = np.arange(len(soil_data), len(soil_data) + 72).reshape(-1, 1)

    soil_gp = GaussianProcessRegressor()
    soil_gp.fit(x, soil_data)
    predicted_soil = soil_gp.predict(predictive_range)

    light_gp = GaussianProcessRegressor()
    light_gp.fit(x, light_data)
    predicted_light = light_gp.predict(predictive_range)
    resp = {
        "soil_observed": soil_data.flatten().tolist(),
        "soil_predicted": predicted_soil.flatten().tolist(),
        "light_observed": light_data.flatten().tolist(),
        "light_predicted": predicted_light.flatten().tolist()
    }

    return json.dumps(resp, indent=4)


if __name__ == "__main__":
    app.run()
