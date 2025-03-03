import numpy as np

# min dim_in = 128 -> 256bit / 16bit
# min dim_out = 8 PIM block
BATCH = 1
# REAL_DIM_IN = 256
# DIM_IN = 256
# DIM_OUT = 1024
REAL_DIM_IN = 1024
DIM_IN = 1024
DIM_OUT = 4096

np.set_printoptions(precision=20)
np.random.seed(1113)

batch_in = np.random.standard_normal(size=(DIM_IN, BATCH)).astype('float16')
for i in range(DIM_IN):
    for j in range(BATCH):
        if i % 2 == 0:
            batch_in[i][j] = 1 + (i / DIM_IN) 
        else:
            batch_in[i][j] = 2 + ((i - 1) / DIM_IN)

for i in range(REAL_DIM_IN, DIM_IN):
    for j in range(0, BATCH):
        batch_in[i][j] = 0

data_w = np.random.standard_normal(size=(DIM_OUT, DIM_IN)).astype('float16')
for i in range(DIM_OUT):
    for j in range(DIM_IN):
        data_w[i][j] = 0
for i in range(DIM_OUT):
    for j in range(DIM_IN):
        # Group each row as groups of 16 elements. Treat each group of 16 
        # elements as a circular buffer, where the position at which
        # the value is written goes up as the indexes increment, and
        # resets when the index reaches the end of the group. 
        if j % 16 == 0:
            data_w[i][16*(j//16)+i%16] = i // DIM_IN + (j / DIM_IN)

# np.random.shuffle(data_w)
batch_out = np.zeros((DIM_OUT, BATCH)).astype('float16')
batch_out = np.matmul(data_w, batch_in)

batch_out2 = np.zeros((DIM_OUT, BATCH)).astype('float16')

for y in range(0, DIM_OUT):
    for x in range(0, DIM_IN):
        batch_out2[y] += data_w[y][x] * batch_in[x][0]

batch_in = batch_in.T.copy()
batch_out = batch_out.T.copy()
batch_out2 = batch_out2.T.copy()

np.save("gemv_input_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_in)
np.save("gemv_weight_" + str(DIM_OUT) + "x" + str(DIM_IN), data_w)
np.save("gemv_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out)
np.save("test_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out2)
with np.printoptions(threshold=np.inf):
    print('batch_in:', batch_in)
    print('data_w:', data_w)
    print('batch_out:', batch_out)
    print('batch_out2:', batch_out2)
    print(batch_in.shape)
    print(data_w.shape)
    print(batch_out.shape)
