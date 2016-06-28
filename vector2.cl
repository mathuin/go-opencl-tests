__kernel void vectAddInt(__global int *A,
			      __global int *B,
			      __global int *C,
			      __private int lenbase)
{
   uint gid = get_global_id(0);
   uint gsize = get_global_size(0);

   C[gid] = A[gid] + B[gid];
}

__kernel void vectSquareUChar(__global uchar *input,
                              __global uchar *output)
{
   size_t id = get_global_id(0);
   output[id] = input[id] * input[id];
}
