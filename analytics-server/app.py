import json
import io

import numpy as np
import boto3
from flask import Flask, request
from flask_cors import CORS, cross_origin
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from object_detector import ObjectDetector

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

dynamo_client = boto3.client('dynamodb', region_name='us-east-2')
s3_client = boto3.client('s3')

detector = ObjectDetector()

@app.route("/")
def index():
    return "Smart Plant API"


@app.route("/health")
def health():
    return 200

@app.route("/sensor")
@cross_origin()
def get_sensor():
    row = str(request.args.get('row'))
    pos = str(request.args.get('pos'))
    resp = dynamo_client.get_item(
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

    soil_kernel = kernels.RationalQuadratic()
    soil_gp = GaussianProcessRegressor(kernel=soil_kernel)
    soil_gp.fit(x, soil_data)
    predicted_soil = soil_gp.predict(predictive_range)

    light_kernel = kernels.RationalQuadratic()
    light_gp = GaussianProcessRegressor(kernel=light_kernel)
    light_gp.fit(x, light_data)
    predicted_light = light_gp.predict(predictive_range)
    resp = {
        "soil_observed": soil_data.flatten().tolist(),
        "soil_predicted": predicted_soil.flatten().tolist(),
        "light_observed": light_data.flatten().tolist(),
        "light_predicted": predicted_light.flatten().tolist()
    }

    return json.dumps(resp, indent=4)

@app.route("/leaves")
@cross_origin()
def get_leaves():
    obj = s3_client.get_object(Bucket='smart-garden', Key='plant-picture.jpg')['Body'].read()
    img = Image.open(io.BytesIO(obj))
    num_leaves, ann_img = detector.detect(img)

    resp = {
        "number_of_leaves": num_leaves,
        "annotated_img": ann_img
    }

    return json.dumps(resp, indent=4)


if __name__ == "__main__":
    app.run()
