import nb_pb2
import nb_pb2_grpc
from concurrent import futures
import logging
import grpc
import time
from classifier import *
import classifier

ONE_DAY_IN_SEC = 12 * 60 * 60


def train_classifier():
    print('Starting to train classifier')
    classifier.train_nb()


class TrainServicer(nb_pb2_grpc.TrainServicer):
    def __init__(self):
        pass

    def TrainClassifier(self, request, context):
        return nb_pb2.TrainReply(message='Classifier trained with '
                                 ' xtrain data:' + request.xtrain + ' ytrain data' + request.ytrain)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nb_pb2_grpc.add_TrainServicer_to_server(
        TrainServicer(), server)
    server.add_insecure_port('[::]:8080')
    server.start()
    try:
        while True:
            print('server listening at 8080')
            time.sleep(ONE_DAY_IN_SEC)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
