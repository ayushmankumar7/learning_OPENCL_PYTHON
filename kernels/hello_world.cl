__kernel void hello_world(__global int* values) {
    int global_id = get_global_id(0);
    printf("Hello World! Got %d from kernel #%d. \n", values[global_id], global_id);
}