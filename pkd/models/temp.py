import tensorflow as tf
import tf2onnx
import onnx

tf_model = tf.keras.models.load_model('/home/aryan.tomar.20031/FaceKD/GN_W1.3_S1_ArcFace_epoch46.h5')
onnx_model = tf2onnx.convert.from_keras(tf_model)
onnx.save_model(onnx_model, 'converted_model.onnx')
