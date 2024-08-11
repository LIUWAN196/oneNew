#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include <time.h>

#include "../../common/nn_common_cpp.h"
#include "../../host/op/relu.h"
#include "../../host/op/maxpool.h"
#include "../../host/op/conv.h"
#include "../../host/op/add.h"
#include "../../host/op/global_avgpool.h"
#include "../../host/op/flatten.h"
#include "../../host/op/gemm.h"

void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
{
	switch (dim.value_case())
	{
	case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
		std::cout << dim.dim_param();
		break;
	case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
		std::cout << dim.dim_value();
		break;
	default:
		assert(false && "should never happen");
	}
}

void print_io_info(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info)
{
	for (auto input_data : info)
	{
		auto shape = input_data.type().tensor_type().shape();
		auto type = input_data.type().tensor_type().elem_type();
		std::cout << " hhhhh  the input data type is: " << type << ":";

		std::cout << "  " << input_data.name() << ":";
		std::cout << "[";
		if (shape.dim_size() != 0)
		{
			int size = shape.dim_size();
			for (int i = 0; i < size - 1; ++i)
			{
				print_dim(shape.dim(i));
				std::cout << ",";
			}
			print_dim(shape.dim(size - 1));
		}
		std::cout << "]\n";
	}
}

void fill_io_cfg(const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &inputs, const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &outputs, char *io_cfg_ptr)
{
	char *cur_io_cfg_ptr = io_cfg_ptr;

	for (auto input : inputs)
	{
		IO_CONFIG_S *io_cfg = (IO_CONFIG_S *)cur_io_cfg_ptr;

		std::string op_type = "io";
		strcpy(io_cfg->op_type, op_type.c_str());

		std::string op_name = "input";
		strcpy(io_cfg->op_name, op_name.c_str());

		strcpy(io_cfg->operand.operand_name, input.name().c_str());
		io_cfg->operand.is_fixed_val = FALSE;

		auto elem_type = input.type().tensor_type().elem_type();
		io_cfg->operand.data_type = (ELEM_TYPE_E)elem_type;

		auto data_shape = input.type().tensor_type().shape();

		if (data_shape.dim_size() == 2)
		{
			io_cfg->operand.shape.N = (int32_t)(data_shape.dim(0).dim_value());
			io_cfg->operand.shape.C = (int32_t)(data_shape.dim(1).dim_value());
		}
		else if (data_shape.dim_size() == 4)
		{
			io_cfg->operand.shape.N = (int32_t)(data_shape.dim(0).dim_value());
			io_cfg->operand.shape.C = (int32_t)(data_shape.dim(1).dim_value());
			io_cfg->operand.shape.H = (int32_t)(data_shape.dim(2).dim_value());
			io_cfg->operand.shape.W = (int32_t)(data_shape.dim(3).dim_value());
		}

		// update cur_io_cfg_ptr
		cur_io_cfg_ptr += align_buf_size(sizeof(IO_CONFIG_S));
	}

	for (auto output : outputs)
	{
		IO_CONFIG_S *io_cfg = (IO_CONFIG_S *)cur_io_cfg_ptr;

		std::string op_type = "io";
		strcpy(io_cfg->op_type, op_type.c_str());

		std::string op_name = "output";
		strcpy(io_cfg->op_name, op_name.c_str());

		strcpy(io_cfg->operand.operand_name, output.name().c_str());
		io_cfg->operand.is_fixed_val = FALSE;

		auto elem_type = output.type().tensor_type().elem_type();
		io_cfg->operand.data_type = (ELEM_TYPE_E)elem_type;

		auto data_shape = output.type().tensor_type().shape();
		if (data_shape.dim_size() == 2)
		{
			io_cfg->operand.shape.N = (int32_t)(data_shape.dim(0).dim_value());
			io_cfg->operand.shape.C = (int32_t)(data_shape.dim(1).dim_value());
			io_cfg->operand.shape.H = (int32_t)(1);
			io_cfg->operand.shape.W = (int32_t)(1);
		}
		else if (data_shape.dim_size() == 4)
		{
			io_cfg->operand.shape.N = (int32_t)(data_shape.dim(0).dim_value());
			io_cfg->operand.shape.C = (int32_t)(data_shape.dim(1).dim_value());
			io_cfg->operand.shape.H = (int32_t)(data_shape.dim(2).dim_value());
			io_cfg->operand.shape.W = (int32_t)(data_shape.dim(3).dim_value());
		}

		// update cur_io_cfg_ptr
		cur_io_cfg_ptr += align_buf_size(sizeof(IO_CONFIG_S));
	}
}

int get_align_cfg_size(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &node, int32_t &node_cnt, int32_t &align_cfg_size)
{
	node_cnt = 0;
	align_cfg_size = 0;

	Manager &m = Manager::getInstance();

	for (auto cur_node : node)
	{
		std::string op_type = cur_node.op_type();

		if (m.Opmap.find(op_type) == m.Opmap.end())
		{
			std::cout << "op: " << op_type << " is not implemented!" << std::endl;
			return -1;
		}

		int a = cfg_size_map[op_type];
		int b = align_buf_size(cfg_size_map[op_type]);
		align_cfg_size += align_buf_size(cfg_size_map[op_type]);
		++node_cnt;
	}

	return 0;
}

void get_align_init_size(const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &init_info, int32_t &init_cnt, int32_t &align_init_size)
{
	init_cnt = 0;
	align_init_size = 0;
	for (auto init_ : init_info)
	{
		auto data_type = init_.data_type();
		auto dims = init_.dims();

		int elem_size = 1;
		int buf_size = 0;

		for (auto dim : dims)
		{
			elem_size *= dim;
		}
		buf_size = elem_size * elem_info_map[data_type].size;

		int c = sizeof(OPERAND_S);

		align_init_size += align_buf_size(sizeof(OPERAND_S) + buf_size);

		++init_cnt;
	}
	return;
}

void fill_node_cfg(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &nodes, char *node_cfg_ptr)
{
	char *cur_node_cfg_ptr = node_cfg_ptr;
	for (auto node : nodes)
	{
		auto op_type = node.op_type();
		auto name = node.name();

		if (op_type == "Relu")
		{
			RELU_CONFIG_S *relu_cfg = (RELU_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(relu_cfg->op_type, op_type.c_str());
			strcpy(relu_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(relu_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(relu_cfg->out_operand_name[out_i++], outp.c_str());
			}
		}
		else if (op_type == "MaxPool")
		{
			MAX_POOL_CONFIG_S *max_pool_cfg = (MAX_POOL_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(max_pool_cfg->op_type, op_type.c_str());

			strcpy(max_pool_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(max_pool_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(max_pool_cfg->out_operand_name[out_i++], outp.c_str());
			}

			// max pool other attributes
			auto attr = node.attribute();
			for (auto param : attr)
			{
				if (param.name() == "ceil_mode")
				{
					for (auto val : param.ints())
					{
						max_pool_cfg->ceil_mode = val;
					}
				}

				if (param.name() == "kernel_shape")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						max_pool_cfg->kernel_shape[i++] = val;
					}
				}

				if (param.name() == "pads")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						max_pool_cfg->pads[i++] = val;
					}
				}

				if (param.name() == "strides")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						max_pool_cfg->strides[i++] = val;
					}
				}
			}
		}
		else if (op_type == "Conv")
		{
			CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(conv_cfg->op_type, op_type.c_str());
			strcpy(conv_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(conv_cfg->in_operand_name[in_i++], inp.c_str());
			}
			if (in_i == 3)
			{
				conv_cfg->has_bias = TRUE;
			}
			else if (in_i == 2)
			{
				conv_cfg->has_bias = FALSE;
			}
			else
			{
				std::cout << "err: the op: " << conv_cfg->op_name << " requires at least 2 inputs, currently only " << in_i << " input." << std::endl;
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(conv_cfg->out_operand_name[out_i++], outp.c_str());
			}

			// Conv other attributes
			auto attr = node.attribute();
			for (auto param : attr)
			{
				if (param.name() == "dilations")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						conv_cfg->dilations[i++] = val;
					}
				}

				if (param.name() == "group")
				{
					for (auto val : param.ints())
					{
						conv_cfg->group = val;
					}
				}

				if (param.name() == "kernel_shape")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						conv_cfg->kernel_shape[i++] = val;
					}
				}

				if (param.name() == "pads")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						conv_cfg->pads[i++] = val;
					}
				}

				if (param.name() == "strides")
				{
					int i = 0;
					for (auto val : param.ints())
					{
						conv_cfg->strides[i++] = val;
					}
				}
			}
		}
		else if (op_type == "Add")
		{
			ADD_CONFIG_S *add_cfg = (ADD_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(add_cfg->op_type, op_type.c_str());
			strcpy(add_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(add_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(add_cfg->out_operand_name[out_i++], outp.c_str());
			}
		}
		else if (op_type == "GlobalAveragePool")
		{
			GLOBAL_AVGPOOL_CONFIG_S *global_avgpool_cfg = (GLOBAL_AVGPOOL_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(global_avgpool_cfg->op_type, op_type.c_str());
			strcpy(global_avgpool_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(global_avgpool_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(global_avgpool_cfg->out_operand_name[out_i++], outp.c_str());
			}
		}
		else if (op_type == "Flatten")
		{
			FLATTEN_CONFIG_S *flatten_cfg = (FLATTEN_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(flatten_cfg->op_type, op_type.c_str());
			strcpy(flatten_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(flatten_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(flatten_cfg->out_operand_name[out_i++], outp.c_str());
			}
		}
		else if (op_type == "Gemm")
		{
			GEMM_CONFIG_S *gemm_cfg = (GEMM_CONFIG_S *)cur_node_cfg_ptr;
			strcpy(gemm_cfg->op_type, op_type.c_str());
			strcpy(gemm_cfg->op_name, name.c_str());

			int in_i = 0;
			for (auto inp : node.input())
			{
				strcpy(gemm_cfg->in_operand_name[in_i++], inp.c_str());
			}

			int out_i = 0;
			for (auto outp : node.output())
			{
				strcpy(gemm_cfg->out_operand_name[out_i++], outp.c_str());
			}
		}

		// update cur_node_cfg_ptr
		cur_node_cfg_ptr += align_buf_size(cfg_size_map[op_type]);
	}
}

void fill_init_info(const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &init_info, char *init_info_ptr)
{

	char *cur_init_info_ptr = init_info_ptr;

	for (auto init_ : init_info)
	{
		OPERAND_S *operand_ptr = (OPERAND_S *)cur_init_info_ptr;
		strcpy(operand_ptr->operand_name, init_.name().c_str());
		operand_ptr->is_fixed_val = TRUE;

		auto data_type = init_.data_type();
		operand_ptr->data_type = (ELEM_TYPE_E)data_type;

		auto dims = init_.dims();
		std::vector<int32_t> vec_dim;
		for (auto dim : dims)
		{
			vec_dim.push_back((int32_t)dim);
		}

		if (vec_dim.size() == 1)
		{
			operand_ptr->shape.N = vec_dim[0];
			operand_ptr->shape.C = 1;
			operand_ptr->shape.H = 1;
			operand_ptr->shape.W = 1;
		}
		else if (vec_dim.size() == 4)
		{
			operand_ptr->shape.N = vec_dim[0];
			operand_ptr->shape.C = vec_dim[1];
			operand_ptr->shape.H = vec_dim[2];
			operand_ptr->shape.W = vec_dim[3];
		}

		auto raw_data = init_.raw_data();
		char *data_ptr = (char *)raw_data.c_str(); // read raw_data
		char *init_data_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
		int init_size = operand_buf_size(operand_ptr);
		memcpy(init_data_ptr, data_ptr, init_size * sizeof(int8_t));

		// update cur_init_info_ptr
		cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
	}
}

int32_t get_time_stamp()
{
	time_t current_time;
	struct tm *local_time;
	current_time = time(NULL);
	local_time = localtime(&current_time);
	int32_t time_stamp = (local_time->tm_year + 1900) * 1000000 + (local_time->tm_mon + 1) * 10000 + local_time->tm_mday * 100 + local_time->tm_hour;

	return time_stamp;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " [onnx_path] [one_path]" << std::endl;
		exit(-1);
	}
	const char *onnx_path = argv[1];
	const char *one_path = argv[2];

	// step 1: load onnx model
	onnx::ModelProto model;
	std::ifstream onnx_file(onnx_path, std::ios::ate | std::ios::binary);
	std::streamsize onnx_size = onnx_file.tellg();
	onnx_file.seekg(0, std::ios::beg);
	std::vector<char> buffer(onnx_size);
	onnx_file.read(buffer.data(), onnx_size);

	// step 2: parse protobuf
	model.ParseFromArray(buffer.data(), onnx_size);
	auto graph = model.graph();

	std::cout << "input\n";
	print_io_info(graph.input());
	std::cout << "output\n";
	print_io_info(graph.output());

	// step 3: get node size of this onnx model, and traverse each node to get op cfg size
	int32_t node_cnt, align_cfg_size;
	if (get_align_cfg_size(graph.node(), node_cnt, align_cfg_size) != 0)
	{
		std::cout << "failed: some op not being implemented temporarily!" << std::endl;
		return -1;
	}

	// step 3: traverse initializer to get init size (including the values of weights and bias)
	int32_t init_cnt, align_init_size;
	get_align_init_size(graph.initializer(), init_cnt, align_init_size);

	// step 3: get in/output op cnt
	int32_t io_cfg_cnt = graph.input().size() + graph.output().size();
	int32_t align_io_cfg_size = io_cfg_cnt * align_buf_size(sizeof(IO_CONFIG_S));

	// step 4: malloc space for .one file
	int head_size = 64; // 64 bytes, to store version information and the num of node and init parameters
	int one_file_size = head_size + align_cfg_size + align_init_size + align_io_cfg_size;
	char *one_file_buf = NULL;
	one_file_buf = (char *)calloc(one_file_size, sizeof(int8_t));

	if (one_file_buf == NULL)
	{
		std::cout << "failed: malloc for one file" << std::endl;
		return -1;
	}

	// step 5: fill the head information
	int32_t *head_info_ptr = (int32_t *)one_file_buf;
	head_info_ptr[0] = get_time_stamp(); // build date
	head_info_ptr[1] = node_cnt;
	head_info_ptr[2] = head_size; // the offset of first node cfg from one_file_buf
	head_info_ptr[3] = init_cnt;
	head_info_ptr[4] = head_size + align_cfg_size; // the offset of first init info from one_file_buf
	head_info_ptr[5] = io_cfg_cnt;
	head_info_ptr[6] = head_size + align_cfg_size + align_init_size; // the offset of first io op cfg from one_file_buf

	// step 6: fill the node cfg
	char *node_cfg_ptr = (char *)(one_file_buf + head_info_ptr[2]);
	fill_node_cfg(graph.node(), node_cfg_ptr);

	// step 7: fill the init info
	char *init_info_ptr = (char *)(one_file_buf + head_info_ptr[4]);
	fill_init_info(graph.initializer(), init_info_ptr);

	// step 7: fill the io cfg
	char *io_cfg_ptr = (char *)(one_file_buf + head_info_ptr[6]);
	fill_io_cfg(graph.input(), graph.output(), io_cfg_ptr);

	// step 8: dump .one file
	FILE *file_p = fopen(one_path, "w");
	fwrite((void *)one_file_buf, 1, one_file_size, file_p);
	fclose(file_p);

	free(one_file_buf);
	onnx_file.close();

	return 0;
}