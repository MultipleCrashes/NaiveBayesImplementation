# Default input for the source is the current directory
protoc --python_out=protobuffpy/ nb.proto

sudo apt-get install protobuff-compiler



nb_pb2.TrainRequest(xtrain='[[1 1 0 1][0 1 0 0][2 2 0 0][2 0 1 0][2 0 1 1][0 0 1 1][1 2 0 0][1 0 1 0][2 2 1 0][1 2 1 1][0 2 0 1][0 1 1 0][2 2 0 1]]',
                            ytrain='[[0][1][1][1][0][1][0][1][1][1][1][1][0]]'))
    print('Response from server ->', response.message)