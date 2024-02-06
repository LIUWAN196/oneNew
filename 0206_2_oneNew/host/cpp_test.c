#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


int32_t load_bin(const char *filename, const int64_t size, char *buf) {
    FILE *file_p = NULL;

    file_p = fopen(filename, "r");
    if (file_p == NULL) {
        printf("cant open the input bin\n");
    }
    size_t bytes_read = fread(buf, sizeof(char), size, file_p);
    fclose(file_p);

    return 0;
}

int32_t write_bin(const char *filename, const int64_t size, char *buf) {
    FILE *file_p = NULL;

    file_p = fopen(filename, "w");

    size_t bytes_written = fwrite((void *) buf, 1, size, file_p);

    fclose(file_p);

    return 0;
}

int32_t submit_task(const char *in_buf, const char *out_buf) {

    printf("hhhhhhh\n");
    void *handle = dlopen("/home/e0006809/Desktop/libadd111.so", RTLD_LAZY);

    int *(*add_liuaa)(const char *, const char *) = (int *(*)()) dlsym(handle, "add_liuaa");
    add_liuaa(in_buf, out_buf);
    printf("hh ese\n");

    return 0;
}


int main(int argc, char **argv) {

    int32_t status;

    printf("start cpp\n");

    if (argc < 2) {
        printf("please into a input.bin\n");
    }

    char *filename = argv[1];
    int64_t input_size = 12;
    char *in_buf = malloc(input_size);
    load_bin(filename, input_size, in_buf);

    int8_t *in_ptr = (int8_t *) in_buf;

    for (int i = 0; i < input_size; ++i) {
        printf("%d ", in_ptr[i]);
    }


    int64_t output_size = 12;
    char *out_buf = malloc(output_size);

    printf("333333333\n");
    status = submit_task(in_buf, out_buf);

    printf("the output is\n");
    int8_t *out_ptr = (int8_t *) out_buf;

    for (int i = 0; i < output_size; ++i) {
        printf("%d ", out_ptr[i]);
    }

    char *out_filename = "output_0.bin";
    write_bin(out_filename, output_size, out_buf);


    return 0;
}
