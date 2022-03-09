import time 
import pyopencl as cl 
import numpy as np 

TASKS = 64 

print("[😁]Loading Program from Kernel")
f = open("kernels/hello_world.cl", 'r', encoding='utf-8')
kernels = ''.join(f.readlines())
f.close()
# print(kernels)
print("[😁]Kernel Read")


print("[😁]Preparing data")
start_time = time.time() 
matrix = np.random.randint(1, 101, dtype=np.int32, size = TASKS)
time_hostdata_loaded = time.time()
# print(matrix)

print("[😁]Creating Context")
context = cl.create_some_context()
print("[😁]Creating Command Queue")
queue = cl.CommandQueue(context, properties = cl.command_queue_properties.PROFILING_ENABLE)
time_ctx_queue_creation = time.time() 

print("[😁]Preparing Device memory for input/output")
dev_matrix = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = matrix)
time_devicedata_loaded = time.time()

print("[😁]Compiling Kernel Code")
program = cl.Program(context, kernels).build()
time_kernel_compilation = time.time()

print("[😁]Execute Kernel Program")
evt = program.hello_world(queue, (TASKS,), (1,), dev_matrix)
print("[😁]Waiting for kernel executions")
evt.wait() 
elapsed = 1e-9 * (evt.profile.end - evt.profile.start)
print("[😁]Done")

print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
print('OpenCL elapsed time          : {}'.format(elapsed))