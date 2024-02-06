#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include "../../common/nn_common_cpp.h"

void show_head_info(char *one_buf_ptr)
{
	int32_t *head_ptr = (int32_t *)one_buf_ptr;
	std::cout << "time stamp: " << head_ptr[0] << ", ";
	std::cout << "node_cnt: " << head_ptr[1] << ", ";
	std::cout << "init_cnt: " << head_ptr[3] << std::endl;
	std::cout << std::endl;
	return;
}

void show_node_cfg_info(char *one_buf_ptr)
{
	int32_t *head_ptr = (int32_t *)one_buf_ptr;
	int32_t node_cnt = head_ptr[1];
	char *cur_node_cfg_ptr = (char *)(one_buf_ptr + head_ptr[2]);

	for (int32_t node_i = 0; node_i < node_cnt; node_i++)
	{
		std::string op_type_str(cur_node_cfg_ptr);
		if (op_type_str == "Relu")
		{
			RELU_CONFIG_S *relu_cfg = (RELU_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: Relu op    ";
			std::cout << "op_name: " << relu_cfg->op_name << ",  ";
			std::cout << "in_operand_name: " << relu_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << relu_cfg->out_operand_name[0] << std::endl;
		}
		else if (op_type_str == "MaxPool")
		{
			MAX_POOL_CONFIG_S *max_pool_cfg = (MAX_POOL_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: MaxPool op    ";
			std::cout << "op_name: " << max_pool_cfg->op_name << ",  ";
			std::cout << "in_operand_name: " << max_pool_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << max_pool_cfg->out_operand_name[0] << ",  ";
			std::cout << "ceil_mode: " << max_pool_cfg->ceil_mode << ",  ";

			int32_t kernel_shape_size = 2;
			std::cout << "kernel_shape: [";
			for (int32_t i = 0; i < kernel_shape_size - 1; i++)
			{
				std::cout << max_pool_cfg->kernel_shape[i] << ", ";
			}
			std::cout << max_pool_cfg->kernel_shape[kernel_shape_size - 1] << "],  ";

			int32_t pads_size = 4;
			std::cout << "pads: [";
			for (int32_t i = 0; i < pads_size - 1; i++)
			{
				std::cout << max_pool_cfg->pads[i] << ", ";
			}
			std::cout << max_pool_cfg->pads[pads_size - 1] << "],  ";

			int32_t strides_size = 2;
			std::cout << "strides: [";
			for (int32_t i = 0; i < strides_size - 1; i++)
			{
				std::cout << max_pool_cfg->strides[i] << ", ";
			}
			std::cout << max_pool_cfg->strides[strides_size - 1] << "]" << std::endl;
		}
		else if (op_type_str == "Conv")
		{
			CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: Conv op    ";
			std::cout << "op_name: " << conv_cfg->op_name << ",  ";

			int32_t in_operand_size = 3;
			if (!conv_cfg->has_bias)
			{
				// this Conv op has no bias
				--in_operand_size;
			}

			std::cout << "in_operand_name: [";
			for (int32_t i = 0; i < in_operand_size - 1; i++)
			{
				std::cout << conv_cfg->in_operand_name[i] << ", ";
			}
			std::cout << conv_cfg->in_operand_name[in_operand_size - 1] << "],  ";

			// std::cout << "in_operand_name: " << conv_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << conv_cfg->out_operand_name[0] << ",  ";
			std::cout << "group: " << conv_cfg->group << ",  ";

			int32_t dilations_size = 2;
			std::cout << "dilations: [";
			for (int32_t i = 0; i < dilations_size - 1; i++)
			{
				std::cout << conv_cfg->dilations[i] << ", ";
			}
			std::cout << conv_cfg->dilations[dilations_size - 1] << "],  ";

			int32_t kernel_shape_size = 2;
			std::cout << "kernel_shape: [";
			for (int32_t i = 0; i < kernel_shape_size - 1; i++)
			{
				std::cout << conv_cfg->kernel_shape[i] << ", ";
			}
			std::cout << conv_cfg->kernel_shape[kernel_shape_size - 1] << "],  ";

			int32_t pads_size = 4;
			std::cout << "pads: [";
			for (int32_t i = 0; i < pads_size - 1; i++)
			{
				std::cout << conv_cfg->pads[i] << ", ";
			}
			std::cout << conv_cfg->pads[pads_size - 1] << "],  ";

			int32_t strides_size = 2;
			std::cout << "strides: [";
			for (int32_t i = 0; i < strides_size - 1; i++)
			{
				std::cout << conv_cfg->strides[i] << ", ";
			}
			std::cout << conv_cfg->strides[strides_size - 1] << "]" << std::endl;
		}
		else if (op_type_str == "Add")
		{
			ADD_CONFIG_S *add_cfg = (ADD_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: Add op    ";
			std::cout << "op_name: " << add_cfg->op_name << ",  ";
			std::cout << "in_operand_name: " << add_cfg->in_operand_name[0] << ",  " << add_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << add_cfg->out_operand_name[0] << std::endl;
		}
		else if (op_type_str == "GlobalAveragePool")
		{
			GLOBAL_AVGPOOL_CONFIG_S *global_avgpool_cfg = (GLOBAL_AVGPOOL_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: GlobalAveragePool op    ";
			std::cout << "op_name: " << global_avgpool_cfg->op_name << ",  ";
			std::cout << "in_operand_name: " << global_avgpool_cfg->in_operand_name[0] << ",  " << global_avgpool_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << global_avgpool_cfg->out_operand_name[0] << std::endl;
		}
		else if (op_type_str == "Flatten")
		{
			FLATTEN_CONFIG_S *flatten_cfg = (FLATTEN_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: Flatten op    ";
			std::cout << "op_name: " << flatten_cfg->op_name << ",  ";
			std::cout << "in_operand_name: " << flatten_cfg->in_operand_name[0] << ",  " << flatten_cfg->in_operand_name[0] << ",  ";
			std::cout << "out_operand_name: " << flatten_cfg->out_operand_name[0] << std::endl;
		}
		else if (op_type_str == "Gemm")
		{
			GEMM_CONFIG_S *gemm_cfg = (GEMM_CONFIG_S *)cur_node_cfg_ptr;
			std::cout << "op_type: Add op    ";
			std::cout << "op_name: " << gemm_cfg->op_name << ",  ";

			int32_t in_operand_size = 3;
			std::cout << "in_operand_name: [";
			for (int32_t i = 0; i < in_operand_size - 1; i++)
			{
				std::cout << gemm_cfg->in_operand_name[i] << ", ";
			}
			std::cout << gemm_cfg->in_operand_name[in_operand_size - 1] << "],  ";

			std::cout << "out_operand_name: " << gemm_cfg->out_operand_name[0] << std::endl;
		}

		std::cout << std::endl;

		// update cur_node_cfg_ptr
		cur_node_cfg_ptr += align_buf_size(cfg_size_map[op_type_str]);
	}

	return;
}

template <typename T>
void show_tensor_value(char *data_ptr, OPERAND_SHAPE_S shape)
{
	T *d_ptr = (T *)data_ptr;
	int32_t n_i, c_i, h_i, w_i;
	for (n_i = 0; n_i < shape.N; n_i++)
	{
		for (c_i = 0; c_i < shape.C; c_i++)
		{
			int32_t offset = n_i * shape.C * shape.H * shape.W + c_i * shape.H * shape.W;
			for (h_i = 0; h_i < shape.H; h_i++)
			{
				for (w_i = 0; w_i < shape.W; w_i++)
				{
					std::cout << d_ptr[offset + h_i * shape.W + w_i] << ",  ";
				}
				std::cout << "      ";
			}
			std::cout << std::endl;
		}
	}

	return;
}

void show_init_info(char *one_buf_ptr, BOOL show_init_value)
{
	int32_t *head_ptr = (int32_t *)one_buf_ptr;
	int32_t init_cnt = head_ptr[3];
	char *cur_init_info_ptr = (char *)(one_buf_ptr + head_ptr[4]);

	for (int32_t init_i = 0; init_i < init_cnt; init_i++)
	{
		OPERAND_S *operand_ptr = (OPERAND_S *)cur_init_info_ptr;
		std::cout << "operand_name: " << operand_ptr->operand_name << ",  ";

		std::cout << "shape: [" << operand_ptr->shape.N << ", " << operand_ptr->shape.C << ", "
				  << operand_ptr->shape.H << ", " << operand_ptr->shape.W << "], ";

		std::cout << "data_type: " << operand_ptr->data_type << std::endl;

		if (show_init_value)
		{
			char *data_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
			if (operand_ptr->data_type == TYPE_FP32)
			{
				show_tensor_value<float>(data_ptr, operand_ptr->shape);
			}
			else if (operand_ptr->data_type == TYPE_INT8)
			{
				show_tensor_value<int8_t>(data_ptr, operand_ptr->shape);
			}
			else if (operand_ptr->data_type == TYPE_INT16)
			{
				show_tensor_value<int16_t>(data_ptr, operand_ptr->shape);
			}
		}

		std::cout << std::endl;

		// update cur_init_info_ptr
		cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + operand_buf_size(operand_ptr));
	}

	return;
}

void show_io_cfg_info(char *one_buf_ptr)
{
	int32_t *head_ptr = (int32_t *)one_buf_ptr;
	int32_t io_cfg_cnt = head_ptr[5];
	char *cur_io_cfg_ptr = (char *)(one_buf_ptr + head_ptr[6]);

	for (int32_t io_cfg_i = 0; io_cfg_i < io_cfg_cnt; io_cfg_i++)
	{
		IO_CONFIG_S *io_cfg_ptr = (IO_CONFIG_S *)cur_io_cfg_ptr;
		std::cout << "op_type: " << io_cfg_ptr->op_type << ",  ";
		std::cout << "op_name: " << io_cfg_ptr->op_name << ",  ";
		std::cout << "operand_name: " << io_cfg_ptr->operand.operand_name << ",  ";

		std::cout << "shape: [" << io_cfg_ptr->operand.shape.N << ", " << io_cfg_ptr->operand.shape.C << ", "
				  << io_cfg_ptr->operand.shape.H << ", " << io_cfg_ptr->operand.shape.W << "], ";

		std::cout << "data_type: " << io_cfg_ptr->operand.data_type << std::endl;

		std::cout << std::endl;

		// update cur_io_cfg_ptr
		cur_io_cfg_ptr += align_buf_size(sizeof(IO_CONFIG_S));
	}

	return;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " [one_path] [optional params: yes]" << std::endl;
		std::cout << "if the second input is \"yes\", it will show_init_value" << std::endl;
		exit(-1);
	}
	const char *one_path = argv[1];
	const char *will_show_init_val = (argc == 3) ? argv[2] : "no";
	std::string will_show_init_val_str(will_show_init_val);

	BOOL show_init_value = FALSE;
	if (will_show_init_val_str == "yes")
	{
		show_init_value = TRUE;
	}

	// step 1: get one file size
	std::ifstream one_file(one_path, std::ios::ate | std::ios::binary);
	int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
	one_file.close();

	// step 2: load one file
	char *one_buf_ptr = (char *)malloc(one_file_size);
	FILE *file_p = NULL;

	file_p = fopen(one_path, "r");
	if (file_p == NULL)
	{
		std::cout << "failed: can't open the one file" << std::endl;
		return 0;
	}
	fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
	fclose(file_p);

	// step 3: show the head information
	std::cout << "=====================================  show_head_info  =====================================" << std::endl;
	show_head_info(one_buf_ptr);

	// step 4: show the node cfg information
	std::cout << "=====================================  show_node_cfg_info  =====================================" << std::endl;
	show_node_cfg_info(one_buf_ptr);

	// step 4: if necessary, show initializer information, such as weight and bias
	std::cout << "=====================================  show_init_info  =====================================" << std::endl;
	show_init_info(one_buf_ptr, show_init_value);

	// step 5: show input output information
	std::cout << "=====================================  show_io_cfg_info  =====================================" << std::endl;
	show_io_cfg_info(one_buf_ptr);

	return 0;
}