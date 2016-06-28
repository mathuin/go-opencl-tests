#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <CL/cl.h>

const char* programSource = 
  "__kernel void vectAddInt(__global int *A,\n"
  "                         __global int *B,\n"
  "                         __global int *C,\n"
  "                         int lenbase)\n"
  "{\n"
  "   uint gid = get_global_id(0);\n"
  "   uint gsize = get_global_size(0);\n"
  "\n"
  "   for (uint idx = gid; idx < lenbase; idx += gsize) {\n"
  "     C[idx] = A[idx] + B[idx];\n"
  "   }\n"
  "}\n";

int main() {
  const int elements = 204800000;

  cl_int status;

  cl_uint numPlatforms = 0;
  cl_platform_id *platforms = NULL;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);

  platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

  status = clGetPlatformIDs(numPlatforms, platforms, NULL);

  for (int p = 0; p < numPlatforms; p++) {

    char *pname = NULL;
    size_t psize;
    clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, psize, pname, &psize);
    pname = (char*)malloc(psize);
    clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, psize, pname, NULL);

    printf("Name: %s\n", pname);

    char *pvendor = NULL;
    clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, psize, pvendor, &psize);
    pvendor = (char*)malloc(psize);
    clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, psize, pvendor, NULL);

    printf("Vendor: %s\n", pvendor);

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

    cl_context context = NULL;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

    for (int d = 0; d < numDevices; d++) {

      char *dname = NULL;
      size_t dsize;
      clGetDeviceInfo(devices[d], CL_DEVICE_NAME, dsize, dname, &dsize);
      dname = (char*)malloc(dsize);
      clGetDeviceInfo(devices[d], CL_DEVICE_NAME, dsize, dname, NULL);

      printf("Name: %s\n", dname);

      char *dvendor = NULL;
      clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, dsize, dvendor, &dsize);
      dvendor = (char*)malloc(dsize);
      clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, dsize, dvendor, NULL);

      printf("Vendor: %s\n", dvendor);

      cl_device_type dtype;
      clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &dtype, NULL);

      switch (dtype) {
      case CL_DEVICE_TYPE_CPU:
	printf("Type: CPU\n");
	break;
      case CL_DEVICE_TYPE_GPU:
	printf("Type: GPU\n");
	break;
      }

      cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
  
      status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

      cl_kernel kernel = NULL;

      kernel = clCreateKernel(program, "vectAddInt", &status);

      cl_command_queue cmdQueue;
      cmdQueue = clCreateCommandQueue(context, devices[d], 0, &status);

      // local_work_size
      // * must be evenly divisible by global_work_size (working it backwards here)
      // * can't clash with __attribute__((reqd_work_group_size(X, Y, Z)))
      // * total number of items across dimensions is greater than CL_DEVICE_MAX_WORK_GROUP_SIZE
      // * items in any dimension exceeed CL_DEVICE_MAX_WORK_ITEM_SIZES in that dimension

      size_t local_work_size;
      local_work_size = elements;

      size_t max_work_group_size;
      clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, dsize, &max_work_group_size, NULL);
      printf("max work group size: %d\n", (int)max_work_group_size);

      if (local_work_size > max_work_group_size) {
	local_work_size = max_work_group_size;
      }

      size_t max_work_item_dimensions;
      clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, dsize, &max_work_item_dimensions, NULL);
      printf("max work item dimensions: %d\n", (int)max_work_item_dimensions);

      size_t max_work_item_sizes[3];
      clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
      printf("max work item sizes[0]: %d\n", (int)max_work_item_sizes[0]);

      if (local_work_size > max_work_item_sizes[0]) {
	local_work_size = max_work_item_sizes[0];
      }

      size_t kernel_work_group_size;
      clGetKernelWorkGroupInfo(kernel, devices[d], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_work_group_size, NULL);
      printf("kernel work group size: %d\n", (int)kernel_work_group_size);

      if (local_work_size > kernel_work_group_size) {
	local_work_size = kernel_work_group_size;
      }

      size_t kernel_preferred_work_group_size_multiple;
      clGetKernelWorkGroupInfo(kernel, devices[d], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &kernel_preferred_work_group_size_multiple, NULL);
      printf("kernel preferred work group size multiple: %d\n", (int)kernel_preferred_work_group_size_multiple);

      local_work_size = (local_work_size/kernel_preferred_work_group_size_multiple)*kernel_preferred_work_group_size_multiple;

      printf("local work size: %d\n", (int)local_work_size);

      // global_work_size 
      // * can't be bigger than 2^(address_bits)-1 (hah)
      // * must be a multiple of local_work_size
      // * largest memory allocation cannot exceed CL_DEVICE_MAX_MEM_ALLOC_SIZE
      // * total memory allocation (dynamic plus static) cannot exceed CL_DEVICE_GLOBAL_MEM_SIZE

      size_t global_work_size;
      global_work_size = elements;

      int static_data, bpe_single, bpe_total;
      // lenbase is the only static element
      static_data = sizeof(int);
      // size of the largest single element
      bpe_single = sizeof(int);
      // total size of one element's allocation
      bpe_total = 3 * bpe_single;
      
      cl_ulong global_mem_size;
      clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);

      if (dtype == CL_DEVICE_TYPE_GPU) {
	// assume less than half the GPU memory is really available
	global_mem_size = (size_t) (0.35 * (float) global_mem_size);
      }
      printf("global mem size: %ld\n", (long)global_mem_size);

      long max_total;
      max_total = (global_mem_size-static_data)/bpe_total;

      if (global_work_size > max_total) {
	global_work_size = max_total;
      }

      cl_ulong max_mem_alloc_size;
      clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
      printf("max mem alloc size: %ld\n", (long)max_mem_alloc_size);

      long max_single;
      max_single = max_mem_alloc_size/bpe_single;

      if (global_work_size > max_single) {
	global_work_size = max_single;
      }

      global_work_size = (global_work_size/local_work_size)*local_work_size;

      // JMT: testing
      printf("global work size: %d\n", (int)global_work_size);

      cl_uint max_compute_units;
      clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, dsize, &max_compute_units, NULL);

      printf("max compute units: %d\n", max_compute_units);

      long multiplier;
      multiplier = global_work_size/(local_work_size*max_compute_units);
      printf("multiplier: %d\n", (int) multiplier);

      int *A = NULL;
      int *B = NULL;
      int *C = NULL;

      size_t datasize = sizeof(int)*elements;

      A = (int*)malloc(datasize);
      B = (int*)malloc(datasize);
      C = (int*)malloc(datasize);

      for (int i = 0; i < elements; i++) {
	A[i] = i;
	B[i] = i;
      }

      // break arrays into sub arrays of 'global_work_size' size
      int Cindex = 0;
      while (Cindex < elements) {
	int sublen;
	sublen = global_work_size;
	if ((elements-Cindex) < global_work_size) {
	  sublen = (elements-Cindex);
	}

	printf("sublen: %d\n", sublen);

	int *subA = NULL;
	int *subB = NULL;
	int *subC = NULL;

	size_t bufsize = sizeof(int)*sublen;
	printf("bufsize: %d\n", (int)bufsize);

	subA = (int*)malloc(bufsize);
	subB = (int*)malloc(bufsize);
	subC = (int*)malloc(bufsize);

	for (int i = 0; i < sublen; i++) {
	  subA[i] = A[Cindex+i];
	  subB[i] = B[Cindex+i];
	}

	cl_mem bufferA;
	cl_mem bufferB;
	cl_mem bufferC;

	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bufsize, NULL, &status);
	if (status != CL_SUCCESS) {
	  printf("create A freak out %d!\n", status);
	}
	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, bufsize, NULL, &status);
	if (status != CL_SUCCESS) {
	  printf("create B freak out %d!\n", status);
	}
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufsize, NULL, &status);
	if (status != CL_SUCCESS) {
	  printf("create C freak out %d!\n", status);
	}

	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_TRUE, 0, bufsize, subA, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
	  printf("write A freak out %d!\n", status);
	}

	status = clFinish(cmdQueue);
	if (status != CL_SUCCESS) {
	  printf("write B freak out %d!\n", status);
	}

	status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_TRUE, 0, bufsize, subB, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
	  printf("write B freak out %d!\n", status);
	}

	status = clFinish(cmdQueue);
	if (status != CL_SUCCESS) {
	  printf("write B freak out %d!\n", status);
	}

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	if (status != CL_SUCCESS) {
	  printf("set A freak out %d!\n", status);
	}
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
	if (status != CL_SUCCESS) {
	  printf("set B freak out %d!\n", status);
	}
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
	if (status != CL_SUCCESS) {
	  printf("set C freak out %d!\n", status);
	}
	status |= clSetKernelArg(kernel, 3, sizeof(cl_int), &sublen);
	if (status != CL_SUCCESS) {
	  printf("set sublen freak out %d!\n", status);
	}

	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	globalWorkSize[0] = sublen;
	localWorkSize[0] = local_work_size;

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
	  printf("kernel freak out %d!\n", status);
	}

	status = clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, bufsize, subC, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
	  printf("read C freak out %d!\n", status);
	}

	// iterate over the output and add it to C
	for (int i = 0; i < sublen; i++) {
	  C[Cindex++] = subC[i];
	}

	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	free(subA);
	free(subB);
	free(subC);
      } // inside iteration

      bool result = true;
      for (int i = 0; i < elements; i++) {
	if (C[i] != i+i) {
	  printf("C[%d] %d != %d\n", i, C[i], i+i);
	  result = false;
	  break;
	}
      }
      if (result) {
	printf("Output is correct\n");
      } else {
	printf("Output is incorrect\n");
	exit(-1);
      }

      free(A);
      free(B);
      free(C);
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(cmdQueue);
      free(dname);
      free(dvendor);
    } // inside devices
    clReleaseContext(context);
    free(devices);
    free(pname);
    free(pvendor);
  } // inside platforms
  free(platforms);
}
