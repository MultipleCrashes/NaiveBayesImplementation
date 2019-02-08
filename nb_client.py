import nb_pb2_grpc
import grpc
import nb_pb2
import numpy as np


def nb_train_classifier_fn(stub):
    response = stub.TrainClassifier(
        nb_pb2.TrainRequest(xtrain='[2, 1, 1, 0, 1, 1, 2]', ytrain='[1]'))
    print('Response from server ->', response.message)


def run():
    with grpc.insecure_channel('localhost:8080') as channel:
        stub = nb_pb2_grpc.TrainStub(channel)
        print('Train the classifier')
        nb_train_classifier_fn(stub)


if __name__ == '__main__':
    run()
