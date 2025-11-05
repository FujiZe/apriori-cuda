# apriori-cuda
cuda Implementation of Apriori Algorithm
# To run the implementation 
1. run "nvcc -DUSE_CUDA apriori.cu -o [ filename ]" for cuda version

2. run "nvcc apriori.cu -o [ filename ]" for no use of cuda

2. then run "./[ filename ] [ min support (Decile) ] transactions_10k.txt output.txt" 