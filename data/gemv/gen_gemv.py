import numpy as np

# min dim_in = 128 -> 256bit / 16bit
# min dim_out = 8 PIM block
BATCH = 1
REAL_DIM_IN = 256
DIM_IN = 256
DIM_OUT = 1024

np.set_printoptions(precision=20)
np.random.seed(1113)

batch_in = np.random.standard_normal(size=(DIM_IN, BATCH)).astype('float16')
for i in range(DIM_IN):
    for j in range(BATCH):
        batch_in[i][j] = i

for i in range(REAL_DIM_IN, DIM_IN):
    for j in range(0, BATCH):
        batch_in[i][j] = 0

data_w = np.random.standard_normal(size=(DIM_OUT, DIM_IN)).astype('float16')
for i in range(DIM_OUT):
    for j in range(DIM_IN):
        if (i % DIM_IN) == j:
            data_w[i][j] = 1 + i // DIM_IN
        else:
            data_w[i][j] = 0

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
