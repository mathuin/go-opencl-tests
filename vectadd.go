package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/mathuin/go-opencl/cl"
)

func main() {

	const elements = 204800011

	for _, platform := range cl.Platforms {
		log.Printf("Platform: %s", platform.Property(cl.PLATFORM_NAME))
		for _, dev := range platform.Devices {
			log.Printf("Device: %s (%s)", dev.Property(cl.DEVICE_NAME), dev.Property(cl.DEVICE_TYPE))
			log.Printf("Version: %s", dev.Property(cl.DEVICE_VERSION))

			var err error
			var context *cl.Context
			var queue *cl.CommandQueue
			var program *cl.Program
			var kernel *cl.Kernel

			if context, err = cl.NewContextOfDevices(nil, []cl.Device{dev}); err != nil {
				panic(err)
			}

			if queue, err = context.NewCommandQueue(dev, cl.QUEUE_NIL); err != nil {
				panic(err)
			}

			if program, err = context.NewProgramFromFile("vector2.cl"); err != nil {
				panic(err)
			}

			if err = program.Build([]cl.Device{dev}, ""); err != nil {
				if status := program.BuildStatus(dev); status != cl.BUILD_SUCCESS {
					panic(fmt.Errorf("Build Error:\n%s\n", program.Property(dev, cl.BUILD_LOG)))
				}
				panic(err)
			}

			if kernel, err = program.NewKernelNamed("vectAddInt"); err != nil {
				panic(err)
			}

			// JMT: testing new kernel functions
			var kernelNumArgs int
			if kernelNumArgs, err = strconv.Atoi(fmt.Sprintf("%d", kernel.Property(cl.KERNEL_NUM_ARGS))); err != nil {
				panic(err)
			}
			log.Printf("kernel num args: %d", kernelNumArgs)

			var localWorkSize int
			if localWorkSize, err = getLocalWorkSize(dev, kernel, elements); err != nil {
				panic(err)
			}

			// this project has one int32 of static data
			staticData := 4
			// each base element is two int32's
			bpeSingle := 2 * 4
			// each retval is one int32
			bpeTotal := bpeSingle + 4

			// global work size
			var globalWorkSize int
			if globalWorkSize, err = getGlobalWorkSize(dev, kernel, elements, staticData, bpeSingle, bpeTotal, localWorkSize); err != nil {
				panic(err)
			}

			var maxComputeUnits int
			if maxComputeUnits, err = strconv.Atoi(fmt.Sprintf("%d", dev.Property(cl.DEVICE_MAX_COMPUTE_UNITS))); err != nil {
				panic(err)
			}
			multiplier := int(float64(globalWorkSize) / (float64(localWorkSize) * float64(maxComputeUnits)))

			log.Printf("multiplier: %d", multiplier)

			A := make([]int32, elements)
			B := make([]int32, elements)
			C := make([]int32, elements)
			for i := int32(0); i < elements; i++ {
				A[i], B[i] = i, i
			}

			var bufA, bufB, bufC *cl.Buffer
			if bufA, err = context.NewBuffer(cl.MEM_READ_ONLY, uint32(globalWorkSize*4)); err != nil {
				panic(err)
			}

			if bufB, err = context.NewBuffer(cl.MEM_READ_ONLY, uint32(globalWorkSize*4)); err != nil {
				panic(err)
			}

			if bufC, err = context.NewBuffer(cl.MEM_WRITE_ONLY, uint32(globalWorkSize*4)); err != nil {
				panic(err)
			}

			if err = kernel.SetArg(0, bufA); err != nil {
				panic(err)
			}

			if err = queue.Finish(); err != nil {
				panic(err)
			}

			if err = kernel.SetArg(1, bufB); err != nil {
				panic(err)
			}

			if err = queue.Finish(); err != nil {
				panic(err)
			}

			if err = kernel.SetArg(2, bufC); err != nil {
				panic(err)
			}

			if err = queue.Finish(); err != nil {
				panic(err)
			}

			Cindex := 0
			for len(A) > 0 {
				sublen := globalWorkSize
				if len(A) < sublen {
					sublen = len(A)
				}
				log.Printf("sublen: %d", sublen)
				var subA []int32
				var subB []int32
				var subC []int32
				subA, A = A[:sublen], A[sublen:]
				subB, B = B[:sublen], B[sublen:]

				var aBytes []byte
				if aBytes, err = getBytes(subA, 0, sublen); err != nil {
					panic(err)
				}
				lenAbytes := uint32(len(aBytes))

				var bBytes []byte
				if bBytes, err = getBytes(subB, 0, sublen); err != nil {
					panic(err)
				}
				lenBbytes := uint32(len(bBytes))

				if lenAbytes != lenBbytes {
					panic(fmt.Errorf("lenAbytes %d != lenBbytes %d", lenAbytes, lenBbytes))
				}

				// JMT: C is just as long as A and B
				lenCbytes := lenAbytes

				if err = queue.EnqueueWriteBuffer(bufA, aBytes, 0); err != nil {
					panic(err)
				}

				if err = queue.Finish(); err != nil {
					panic(err)
				}

				if err = queue.EnqueueWriteBuffer(bufB, bBytes, 0); err != nil {
					panic(err)
				}

				if err = queue.Finish(); err != nil {
					panic(err)
				}

				if err = kernel.SetArg(3, int32(lenCbytes)); err != nil {
					panic(err)
				}

				if err = queue.Finish(); err != nil {
					panic(err)
				}

				if err = queue.EnqueueKernel(kernel, []cl.Size{0}, []cl.Size{cl.Size(globalWorkSize)}, []cl.Size{cl.Size(localWorkSize)}); err != nil {
					panic(err)
				}

				if err = queue.Finish(); err != nil {
					panic(err)
				}

				if outBuf, err := queue.EnqueueReadBuffer(bufC, 0, lenCbytes); err != nil {
					panic(err)
				} else {
					if subC, err = getInt32s(outBuf, 0, sublen); err != nil {
						panic(err)
					}
					copy(C[Cindex:], subC)
					Cindex += sublen
				}
			}
			// test
			for i := int32(0); i < elements; i++ {
				if C[i] != i<<1 {
					log.Fatalf("Output at i=%d is incorrect: %d != %d", i, C[i], i<<1)
				}
			}
		}
	}
}

// get_local_work_size calculates the best local work size for a given kernel based on the device's limitations.
// The argument max represents the maximum possible value for local work size -- the number of elements in the whole run.
func getLocalWorkSize(dev cl.Device, kernel *cl.Kernel, max int) (localWorkSize int, err error) {
	localWorkSize = max

	var maxWorkGroupSize int
	if maxWorkGroupSize, err = strconv.Atoi(fmt.Sprintf("%d", dev.Property(cl.DEVICE_MAX_WORK_GROUP_SIZE))); err != nil {
		return 0, err
	}
	log.Printf("max work group size: %d", maxWorkGroupSize)

	if localWorkSize > maxWorkGroupSize {
		localWorkSize = maxWorkGroupSize
	}

	var workItemDimensions int
	if workItemDimensions, err = strconv.Atoi(fmt.Sprintf("%d", dev.Property(cl.DEVICE_MAX_WORK_ITEM_DIMENSIONS))); err != nil {
		return 0, err
	}
	log.Printf("work item dimensions: %d", workItemDimensions)

	var maxWorkItemSizes []int
	maxWorkItemSizesStr := fmt.Sprintf("%v", dev.Property(cl.DEVICE_MAX_WORK_ITEM_SIZES))
	for _, v := range strings.Split(maxWorkItemSizesStr[1:len(maxWorkItemSizesStr)-1], " ") {
		var val int
		if val, err = strconv.Atoi(v); err != nil {
			return 0, err
		}
		maxWorkItemSizes = append(maxWorkItemSizes, val)
	}

	if len(maxWorkItemSizes) != workItemDimensions {
		return 0, fmt.Errorf("length of max_work_item_sizes %d != work_item_dimensions %d", len(maxWorkItemSizes), workItemDimensions)
	}

	log.Printf("max work item sizes: %v", maxWorkItemSizes)

	if localWorkSize > maxWorkItemSizes[0] {
		localWorkSize = maxWorkItemSizes[0]
	}

	var kernelWorkGroupSize int
	if kernelWorkGroupSize, err = strconv.Atoi(fmt.Sprintf("%d", kernel.WorkGroupProperty(dev, cl.KERNEL_WORK_GROUP_SIZE))); err != nil {
		return 0, err
	}
	log.Printf("kernel work group size: %d", kernelWorkGroupSize)

	if localWorkSize > kernelWorkGroupSize {
		localWorkSize = kernelWorkGroupSize
	}

	var kernelPreferredWorkGroupSizeMultiple int
	if kernelPreferredWorkGroupSizeMultiple, err = strconv.Atoi(fmt.Sprintf("%d", kernel.WorkGroupProperty(dev, cl.PREFERRED_WORK_GROUP_SIZE_MULTIPLE))); err != nil {
		return 0, err
	}
	log.Printf("kernel preferred work group size multiple: %d", kernelPreferredWorkGroupSizeMultiple)

	localWorkSize = int(float64(localWorkSize)/float64(kernelPreferredWorkGroupSizeMultiple)) * kernelPreferredWorkGroupSizeMultiple

	log.Printf("local work size: %d", localWorkSize)

	return localWorkSize, nil
}

// get_global_work_size calculates the best global work size for a given kernel based on the device's limitations and the sizes of the various bits of data used in the algorithm.
// The argument max represents the maximum possible value for local work size -- the number of elements in the whole run.
func getGlobalWorkSize(dev cl.Device, kernel *cl.Kernel, max int, staticData int, bpeSingle int, bpeTotal int, localWorkSize int) (globalWorkSize int, err error) {
	globalWorkSize = max

	var globalMemSize int
	if globalMemSize, err = strconv.Atoi(fmt.Sprintf("%d", dev.Property(cl.DEVICE_GLOBAL_MEM_SIZE))); err != nil {
		return 0, err
	}

	log.Printf("global mem size: %d", globalMemSize)

	maxTotal := int((float64(globalMemSize) - float64(staticData)) / float64(bpeTotal))

	if globalWorkSize > maxTotal {
		log.Printf("reducing global_work_size from %d to %d", globalWorkSize, maxTotal)
		globalWorkSize = maxTotal
	}

	var memAllocSize int
	if memAllocSize, err = strconv.Atoi(fmt.Sprintf("%d", dev.Property(cl.DEVICE_MAX_MEM_ALLOC_SIZE))); err != nil {
		return 0, err
	}
	log.Printf("max mem alloc size: %d", memAllocSize)

	maxSingle := int(float64(memAllocSize) / float64(bpeSingle))
	if globalWorkSize > maxSingle {
		log.Printf("reducing global_work_size from %d to %d", globalWorkSize, maxSingle)
		globalWorkSize = maxSingle
	}

	globalWorkSize = int(float64(globalWorkSize)/float64(localWorkSize)) * localWorkSize

	log.Printf("global work size: %d", globalWorkSize)

	return globalWorkSize, nil
}

// get_bytes converts an array of int32's into a byte array.
func getBytes(arr []int32, offset int, len int) (out []byte, err error) {
	aBuffer := new(bytes.Buffer)
	for i := offset; i < offset+len; i++ {
		if err = binary.Write(aBuffer, binary.LittleEndian, arr[i]); err != nil {
			return nil, err
		}
	}
	return aBuffer.Bytes(), nil
}

// get_int32s converts a byte byte into an array of int32's.
func getInt32s(out []byte, offset int, len int) (arr []int32, err error) {
	cBuffer := bytes.NewReader(out)
	var elem int32
	// JMT: THIS IS HORRIBLE
	for i := 0; i < offset+len; i++ {
		if err = binary.Read(cBuffer, binary.LittleEndian, &elem); err != nil {
			return nil, err
		}
		if i >= offset {
			arr = append(arr, elem)
		}
	}
	return arr, nil
}
