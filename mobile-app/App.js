import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import { RNS3 } from 'react-native-aws3';
let camera

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    const photo = await camera.takePictureAsync();
    const uri = photo.uri;
    let file = {
      uri: uri,
      name: "plant-picture.jpg",
      type: "image/jpg"
    }
    let options = { encoding: FileSystem.EncodingType.Base64 };
    let fileString = await FileSystem.readAsStringAsync(uri, options);
    const s3Options = {
      bucket: "smart-garden",
      region: "us-east-2",
      accessKey: process.env["AWS_ACCESS_KEY"],
      secretKey: process.env["AWS_SECRET_KEY"],
      successActionStatus: 201
    }
    RNS3.put(file, s3Options).then(resp => {
      if (resp.status !== 201)
        console.log(resp.status)
        throw new Error("Failed to upload image to S3");

      console.log(resp.body);
    }).catch(e => console.log(e))
    Alert.alert("Success");
    let base64String = 'data:image/jpg;base64' + fileString;
  }

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  return (
    <View style={{ flex: 1 }}>
      <View style={{ flex: 1 }}>
        <Camera style={{ flex: 1 }} type={type} ref={(r) => {
          camera = r
        }}>
          <View
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              flexDirection: 'row',
              justifyContent: 'center'
            }}>
            <TouchableOpacity
              style={{
                flex: 0.4,
                alignSelf: 'flex-end',
                alignItems: 'center',
                backgroundColor: 'white',
                borderRadius: 50,
                marginBottom: 40
              }}
              onPress={() => takePicture()}>
              <Text style={{ fontSize: 18, margin: 10, color: '#32CD32', fontWeight: 'bold' }}> Capture </Text>
            </TouchableOpacity>
          </View>
        </Camera>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
