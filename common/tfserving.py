# coding=utf8
import sys

import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf

channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

tfrecord_path = "/part-r-00000.gz"
examples = []
options = tf.io.TFRecordOptions(compression_type="GZIP")
for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path, options=options):
    examples.append(serialized_example)


def do_request_v2(proto, stub, run_mode='pred'):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "model"
    request.model_spec.signature_name = run_mode
    request.inputs['input'].CopyFrom(tf.compat.v1.make_tensor_proto(proto))
    request.inputs['record_type'].CopyFrom(tf.compat.v1.make_tensor_proto("Example"))
    rsp = stub.Predict(request)
    return rsp


res = do_request_v2(examples, stub)
print(res)




#pred_ids = np.squeeze(tf.make_ndarray(res['pred_ids']))










