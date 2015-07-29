all: opencl_experiments

opencl_experiments: main.c
	gcc main.c -lOpenCL -oopencl_experiments -Wdeclaration-after-statement -std=c89

clean:
	rm -f opencl_experiments
