
import tensorflow as tf

# Load your saved model
model = tf.saved_model.load("modelv4")

# Print the graph operations
for op in model.signatures["serving_default"].graph.get_operations():
    print(op.name)


print()
def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


printTensors("modelv4")

