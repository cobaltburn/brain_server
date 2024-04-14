import tensorflow as tf

model_path = "modelv4/saved_model.pb"

with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
