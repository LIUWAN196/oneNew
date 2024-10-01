#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include <time.h>

#include "../manager/manager.h"

#include "ops_head.h"
#include "net.h"

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
        LOG_ERR("should never happen");
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

        io_cfg->operand.dim_num_of_shapes = data_shape.dim_size();
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            io_cfg->operand.shapes[dim_i] = 1;
        }

        for (int i = 0; i < data_shape.dim_size(); ++i) {
            io_cfg->operand.shapes[i] = (int32_t)(data_shape.dim(i).dim_value());
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
        io_cfg->operand.dim_num_of_shapes = data_shape.dim_size();
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            io_cfg->operand.shapes[dim_i] = 1;
        }
//        memset(& io_cfg->operand.shapes[0], 1, SHAPE_LEN * sizeof(int32_t));
        for (int i = 0; i < data_shape.dim_size(); ++i) {
            io_cfg->operand.shapes[i] = (int32_t)(data_shape.dim(i).dim_value());
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
        if (op_type == "Constant" || op_type == "Identity" ) {
            continue; // Constant op will be processed elsewhere
        }

		if (m.Opmap.find(op_type) == m.Opmap.end())
		{
            LOG_ERR("sorry, op: %s is not implemented!", op_type.c_str());
			return -1;
		}

		align_cfg_size += align_buf_size(m.op_cfg_size_map[op_type]);
		++node_cnt;
	}

	return 0;
}

void get_align_init_size(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &nodes,
                         const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &init_info,
                         int32_t &init_cnt, int32_t &align_init_size)
{
	init_cnt = 0;
	align_init_size = 0;

    // 这里将 constant 算子的输出也作为 init 数据，为其准备空间
    for (auto node : nodes) {
        auto op_type = node.op_type();
        auto name = node.name();

        if (op_type == "Constant") {
            for (auto attr: node.attribute()) {
                if (attr.name() == "value") {
                    auto constant_tensor = attr.t();
                    auto data_type = constant_tensor.data_type();
                    auto dims = constant_tensor.dims();

                    int elem_size = 1;
                    for (auto dim : dims)
                    {
                        elem_size *= dim;
                    }
                    int buf_size = elem_size * elem_info_map[data_type].size;

                    align_init_size += align_buf_size(sizeof(OPERAND_S) + buf_size);

                    ++init_cnt;
                }
            }
        } // end this Constant op
    }

    // 为 init 数据准备空间
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

        align_init_size += align_buf_size(sizeof(OPERAND_S) + buf_size);

		++init_cnt;
	}
	return;
}

void fill_node_cfg(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &nodes,
                   char *node_cfg_ptr, int32_t init_cnt, char *init_info_ptr)
{
    Manager &m = Manager::getInstance();

    // 遍历 init_info_ptr，以 operand_name 为键，OPERAND_S + 后续数据为值，放到 map 中
    char *cur_init_info_ptr = init_info_ptr;
    std::unordered_map<std::string, OPERAND_S*> init_info_map;
    for (int i = 0; i < init_cnt; ++i) {
        OPERAND_S *operand_ptr = (OPERAND_S *) cur_init_info_ptr;
        std::string init_operands = std::string(operand_ptr->operand_name);
        init_info_map[init_operands] = operand_ptr;

        // update cur_init_info_ptr
        int init_size = operand_buf_size(operand_ptr);
        cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
    }


    // 开始填充 node cfg
    char *cur_node_cfg_ptr = node_cfg_ptr;
	for (auto node : nodes) {
        auto op_type = node.op_type();
        auto name = node.name();

        if (op_type == "Constant")
        {
            // 在这里不用处理，已经在 get_align_init_size 和 fill_init_info 这两个函数中处理为 init 数据了
            continue;
        }

        BASE_CONFIG_S *base_cfg = (BASE_CONFIG_S *)cur_node_cfg_ptr;
        strcpy(base_cfg->op_type, op_type.c_str());
        strcpy(base_cfg->op_name, name.c_str());

        int in_i = 0;
        for (auto inp : node.input())
        {
            strcpy(base_cfg->in_operand_name[in_i++], inp.c_str());
        }
        base_cfg->in_operand_num = in_i;

        int out_i = 0;
        for (auto outp : node.output())
        {
            strcpy(base_cfg->out_operand_name[out_i++], outp.c_str());
        }
        base_cfg->out_operand_num = out_i;

        if (op_type == "ArgMax")
        {
            ARGMAX_CONFIG_S *argmax_cfg = (ARGMAX_CONFIG_S *)cur_node_cfg_ptr;

            // max pool other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    argmax_cfg->axis = param.i();
                } else if (param.name() == "keepdims")
                {
                    argmax_cfg->keepdims = param.i();
                } else if (param.name() == "select_last_index")
                {
                    argmax_cfg->select_last_index = param.i();
                }
            }
        }
        else if (op_type == "AveragePool")
        {
            AVG_POOL_CONFIG_S *avg_pool_cfg = (AVG_POOL_CONFIG_S *)cur_node_cfg_ptr;

            // max pool other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "ceil_mode")
                {
                    avg_pool_cfg->ceil_mode = param.i();
                }

                if (param.name() == "kernel_shape")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        avg_pool_cfg->kernel_shape[i++] = val;
                    }
                }

                if (param.name() == "pads")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        avg_pool_cfg->pads[i++] = val;
                    }
                }

                if (param.name() == "strides")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        avg_pool_cfg->strides[i++] = val;
                    }
                }
            }
        }
        else if (op_type == "Clip")
        {
            CLIP_CONFIG_S *clip_cfg = (CLIP_CONFIG_S *)cur_node_cfg_ptr;

            // other attributes
            // first in_operand is ifmap; second is min; third is max
            std::string min_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S* min_operand = init_info_map[min_name];
            if (min_operand == NULL) {
                clip_cfg->min = -1 * FLT_MAX;
            } else {
                float * min_data_ptr = (float*)((char*)min_operand + sizeof(OPERAND_S));
                clip_cfg->min = min_data_ptr[0];
            }

            std::string max_name = std::string(base_cfg->in_operand_name[2]);
            OPERAND_S* max_operand = init_info_map[max_name];
            if (max_operand == NULL) {
                clip_cfg->max = FLT_MAX;
            } else {
                float * max_data_ptr = (float*)((char*)max_operand + sizeof(OPERAND_S));
                clip_cfg->max = max_data_ptr[0];
            }

//            LOG_DBG("clip_cfg->min is %f, clip_cfg->max is %f", clip_cfg->min, clip_cfg->max);

            // clip 算子的第 2、3 个输入是 min 和 max，这个我放到 cfg 中，不作为输入。所以这里修改 clip 算子的输入为 1 个
            clip_cfg->op_base_cfg.in_operand_num = 1;
            memset(clip_cfg->op_base_cfg.in_operand_name[1], 0, (OPERAND_MAXNUM - 1) * OPERAND_NAME_LEN);

        }
        else if (op_type == "Concat")
        {
            CONCAT_CONFIG_S *concat_cfg = (CONCAT_CONFIG_S *)cur_node_cfg_ptr;

            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    concat_cfg->axis = param.i();

                }
            }
        }
        else if (op_type == "Conv")
        {
            CONV_CONFIG_S *conv_cfg = (CONV_CONFIG_S *)cur_node_cfg_ptr;

            int in_i = base_cfg->in_operand_num;

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
                LOG_ERR("this op is %s, requires at least 2 inputs, currently only %d input.", base_cfg->op_name, in_i);
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
                    conv_cfg->group = param.i();
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
        else if (op_type == "ConvTranspose")
        {
            CONV_TRANSPOSE_CONFIG_S *conv_transpose_cfg = (CONV_TRANSPOSE_CONFIG_S *)cur_node_cfg_ptr;

            int in_i = base_cfg->in_operand_num;

            if (in_i == 3)
            {
                conv_transpose_cfg->has_bias = TRUE;
            }
            else if (in_i == 2)
            {
                conv_transpose_cfg->has_bias = FALSE;
            }
            else
            {
                LOG_ERR("this op is %s, requires at least 2 inputs, currently only %d input.", base_cfg->op_name, in_i);
            }

            // conv_transpose_cfg other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "dilations")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        conv_transpose_cfg->dilations[i++] = val;
                    }
                }

                if (param.name() == "group")
                {
                    conv_transpose_cfg->group = param.i();
                }

                if (param.name() == "kernel_shape")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        conv_transpose_cfg->kernel_shape[i++] = val;
                    }
                }

                if (param.name() == "pads")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        conv_transpose_cfg->pads[i++] = val;
                    }
                }

                if (param.name() == "strides")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        conv_transpose_cfg->strides[i++] = val;
                    }
                }
            }
        }
        else if (op_type == "Einsum")
        {
            EINSUM_CONFIG_S *einsum_cfg = (EINSUM_CONFIG_S *)cur_node_cfg_ptr;

            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "equation")
                {
                    std::string equation = param.s();
                    if (equation != "bmchw,bnmc->bmhwn" && equation != "bchw,bkc->bkhw") {
                        LOG_ERR("sorry, cur, the Einsum's equation must be bmchw,bnmc->bmhwn or bchw,bkc->bkhw");
                    }
                    strcpy(einsum_cfg->equation, equation.c_str());
//                    std::cout << "einsum op's equation is: " << param.s() << std::endl;
//                    LOG_DBG("einsum op's equation is:%s", param.s());
                }
            }
        }
        else if (op_type == "Expand")
        {
            EXPAND_CONFIG_S *expand_cfg = (EXPAND_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;

            // first in_operand is ifmap; second is dst shape
            std::string dst_shape_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S* dst_shape_operand = init_info_map[dst_shape_name];
            int64_t* dst_shape_data_ptr = (int64_t*)((char*)dst_shape_operand + sizeof(OPERAND_S));

            expand_cfg->dst_shape_num = operand_elem_size(dst_shape_operand);
            for (int i = 0; i < expand_cfg->dst_shape_num; ++i) {
                expand_cfg->dst_shape[i] = dst_shape_data_ptr[i];
            }
            expand_cfg->dst_shape_num = dst_shape_operand->shapes[0];

        }
        else if (op_type == "Flatten")
        {
            FLATTEN_CONFIG_S *flatten_cfg = (FLATTEN_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;

            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    flatten_cfg->axis = param.i();
                }
            }

        }
        else if (op_type == "GridSample")
        {
            GRID_SAMPLE_CONFIG_S *grid_sample_cfg = (GRID_SAMPLE_CONFIG_S *)cur_node_cfg_ptr;

            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "align_corners")
                {
                    grid_sample_cfg->align_corners = param.i();
                }
                if (param.name() == "mode")
                {
                    std::string mode = param.s();
                    if (mode != "bilinear") {
                        LOG_ERR("sorry, cur, the GridSample's mode must be bilinear");
                    }
                    strcpy(grid_sample_cfg->mode, mode.c_str());
                }
                if (param.name() == "padding_mode")
                {
                    std::string padding_mode = param.s();
                    if (padding_mode != "zeros") {
                        LOG_ERR("sorry, cur, the GridSample's padding_mode must be zeros");
                    }
                    strcpy(grid_sample_cfg->padding_mode, padding_mode.c_str());
                }
            }
        }
        else if (op_type == "Gather")
        {
            GATHER_CONFIG_S *gather_cfg = (GATHER_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;   //  indices trans to attr

            // other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    gather_cfg->axis = param.i();
                }
            }

            // first in_operand is ifmap; second is indices
            std::string indices_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S* indices_operand = init_info_map[indices_name];
            if (indices_operand != NULL) {
                // gather 算子的索引只有一个，并且是 initial 属性的，所以直接放到 cfg 中
                gather_cfg->indices_from_ifmap = FALSE;
                int64_t * indices_data_ptr = (int64_t*)((char*)indices_operand + sizeof(OPERAND_S));
                gather_cfg->indices = indices_data_ptr[0];
                int a = 101;
            } else {
                // gather 算子的索引不止一个，而且是变量，会随着输入改变而改变，所以是作为输出传进来的
                gather_cfg->indices_from_ifmap = TRUE;
                base_cfg->in_operand_num = 2;   //  data and indices
            }
        }
        else if (op_type == "HardSigmoid")
        {
            HARD_SIGMOID_CONFIG_S *hard_sigmoid_cfg = (HARD_SIGMOID_CONFIG_S *)cur_node_cfg_ptr;

            // other attributes
            hard_sigmoid_cfg->alpha = 0.2f; // alpha default is '0.2'
            hard_sigmoid_cfg->beta = 0.5f;  // beta default is '0.5'
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "alpha")
                {
                    hard_sigmoid_cfg->alpha = param.f();
                } else if (param.name() == "beta")
                {
                    hard_sigmoid_cfg->beta = param.f();
                }
            }
        }
        else if (op_type == "LeakyRelu")
        {
            LEAKYRELU_CONFIG_S *leaky_relu_cfg = (LEAKYRELU_CONFIG_S *)cur_node_cfg_ptr;

            // other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "alpha")
                {
                    leaky_relu_cfg->alpha = param.f();
                }

            }
        }
        else if (op_type == "MaxPool")
		{
			MAX_POOL_CONFIG_S *max_pool_cfg = (MAX_POOL_CONFIG_S *)cur_node_cfg_ptr;

			// max pool other attributes
			auto attr = node.attribute();
			for (auto param : attr)
			{
				if (param.name() == "ceil_mode")
				{
                    max_pool_cfg->ceil_mode = param.i();
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
        else if (op_type == "Pad")
        {
            PAD_CONFIG_S *pad_cfg = (PAD_CONFIG_S *)cur_node_cfg_ptr;

            std::string pads_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S* pads_operand = init_info_map[pads_name];
            int64_t * pads_data_ptr = (int64_t*)((char*)pads_operand + sizeof(OPERAND_S));
            const int pads_num = 8;
            for (int i = 0; i < pads_num; ++i) {
                pad_cfg->pads[i] = pads_data_ptr[i];
            }
            pad_cfg->op_base_cfg.in_operand_num = 1;

        }
        else if (op_type == "ReduceMax")
        {
            REDUCE_MAX_CONFIG_S *reduce_max_cfg = (REDUCE_MAX_CONFIG_S *)cur_node_cfg_ptr;

            // other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axes")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        reduce_max_cfg->axes[i++] = val;
                    }
                    reduce_max_cfg->axes_num = i;
                }
                reduce_max_cfg->keepdims = 1;
                if (param.name() == "keepdims")
                {
                    reduce_max_cfg->keepdims = param.i();
                }
            }
        }
        else if (op_type == "ReduceMean")
        {
            REDUCE_MEAN_CONFIG_S *reduce_mean_cfg = (REDUCE_MEAN_CONFIG_S *)cur_node_cfg_ptr;

            // other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axes")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        reduce_mean_cfg->axes[i++] = val;
                    }
                    reduce_mean_cfg->axes_num = i;
                }
                reduce_mean_cfg->keepdims = 1;
                if (param.name() == "keepdims")
                {
                    reduce_mean_cfg->keepdims = param.i();
                }
            }
        }
        else if (op_type == "ReduceSum")
        {
            REDUCE_SUM_CONFIG_S *reduce_sum_cfg = (REDUCE_SUM_CONFIG_S *)cur_node_cfg_ptr;

            // max pool other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "keepdims")
                {
                    reduce_sum_cfg->keepdims = param.i();
                } else if (param.name() == "noop_with_empty_axes")
                {
                    reduce_sum_cfg->noop_with_empty_axes = param.i();
                }
            }
            reduce_sum_cfg->op_base_cfg.in_operand_num = 1;
        }
        else if (op_type == "Resize")
        {
            RESIZE_CONFIG_S *resize_cfg = (RESIZE_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;   //  such as roi and scales trans to attr

            // other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "cubic_coeff_a")
                {
                    resize_cfg->cubic_coeff_a = param.f();
                }
            }

            // first in_operand is ifmap; second is roi; third is scales; fourth is sizes
            std::string resize_scales_name = std::string(base_cfg->in_operand_name[2]);
            OPERAND_S* scales_operand = init_info_map[resize_scales_name];
            std::string resize_sizes_name = std::string(base_cfg->in_operand_name[3]);
            OPERAND_S* sizes_operand = init_info_map[resize_sizes_name];
            if ((scales_operand == NULL || operand_elem_size(scales_operand) == 0) && sizes_operand != NULL) {
                // using sizes
                int64_t * sizes_data_ptr = (int64_t*)((char*)sizes_operand + sizeof(OPERAND_S));
                resize_cfg->scales_num = operand_elem_size(sizes_operand);
                for (int i = 0; i < resize_cfg->scales_num; ++i) {
                    resize_cfg->sizes[i] = sizes_data_ptr[i];
                }
            } else if (scales_operand != NULL) {
                float * scales_data_ptr = (float*)((char*)scales_operand + sizeof(OPERAND_S));
                resize_cfg->scales_num = operand_elem_size(scales_operand);
                for (int i = 0; i < resize_cfg->scales_num; ++i) {
                    resize_cfg->scales[i] = scales_data_ptr[i];
                }
            } else {
                resize_cfg->scales_num = 4;
                resize_cfg->scales[0] = 1;
                resize_cfg->scales[1] = 1;
                resize_cfg->scales[2] = 2;
                resize_cfg->scales[3] = 2;
            }
        }
        else if (op_type == "Reshape")
        {
            RESHAPE_CONFIG_S *reshape_cfg = (RESHAPE_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;

            // first in_operand is ifmap; second is dst shape
            std::string dst_shape_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S* dst_shape_operand = init_info_map[dst_shape_name];
            int64_t * dst_shape_data_ptr = (int64_t*)((char*)dst_shape_operand + sizeof(OPERAND_S));

            reshape_cfg->dst_shape_num = operand_elem_size(dst_shape_operand);
            for (int i = 0; i < reshape_cfg->dst_shape_num; ++i) {
                reshape_cfg->dst_shape[i] = dst_shape_data_ptr[i];
            }
        }
        else if (op_type == "Slice")
        {
            SLICE_CONFIG_S *slice_cfg = (SLICE_CONFIG_S *)cur_node_cfg_ptr;

            // first in_operand is ifmap; second is starts; third is ends; fourth is axes; five is steps
            {
                std::string init_name = std::string(base_cfg->in_operand_name[1]);
                OPERAND_S* init_operand = init_info_map[init_name];
                int64_t * init_data_ptr = (int64_t*)((char*)init_operand + sizeof(OPERAND_S));

                slice_cfg->slice_axes_num = operand_elem_size(init_operand);
                for (int i = 0; i <  slice_cfg->slice_axes_num; ++i) {
                    slice_cfg->starts[i] = init_data_ptr[i];
                }
            }

            {
                std::string init_name = std::string(base_cfg->in_operand_name[2]);
                OPERAND_S* init_operand = init_info_map[init_name];
                int64_t * init_data_ptr = (int64_t*)((char*)init_operand + sizeof(OPERAND_S));

                slice_cfg->slice_axes_num = operand_elem_size(init_operand);
                for (int i = 0; i <  slice_cfg->slice_axes_num; ++i) {
                    slice_cfg->ends[i] = init_data_ptr[i];
                }
            }

            {
                std::string init_name = std::string(base_cfg->in_operand_name[3]);
                OPERAND_S* init_operand = init_info_map[init_name];
                int64_t * init_data_ptr = (int64_t*)((char*)init_operand + sizeof(OPERAND_S));

                slice_cfg->slice_axes_num = operand_elem_size(init_operand);
                for (int i = 0; i <  slice_cfg->slice_axes_num; ++i) {
                    slice_cfg->axes[i] = init_data_ptr[i];
                }
            }

            {
                std::string init_name = std::string(base_cfg->in_operand_name[4]);
                OPERAND_S* init_operand = init_info_map[init_name];
                if (init_operand == NULL) {
                    for (int i = 0; i <  SHAPE_LEN; ++i) {
                        slice_cfg->steps[i] = 1;
                    }
                } else {
                    int64_t * init_data_ptr = (int64_t*)((char*)init_operand + sizeof(OPERAND_S));

                    slice_cfg->slice_axes_num = operand_elem_size(init_operand);
                    for (int i = 0; i <  slice_cfg->slice_axes_num; ++i) {
                        slice_cfg->steps[i] = init_data_ptr[i];
                    }
                }

            }

            base_cfg->in_operand_num = 1;   //  slice other(such as starts ends) operand have been trans to attr

        }
        else if (op_type == "Softmax")
        {
            SOFTMAX_CONFIG_S *softmax_cfg = (SOFTMAX_CONFIG_S *)cur_node_cfg_ptr;

            // max pool other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    softmax_cfg->axis = param.i();
                }
            }
        }
        else if (op_type == "Split")
        {
            SPLIT_CONFIG_S *split_cfg = (SPLIT_CONFIG_S *)cur_node_cfg_ptr;

            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "axis")
                {
                    split_cfg->axis = param.i();
                }

                if (param.name() == "split")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        split_cfg->split[i++] = val;
                    }
                    split_cfg->split_num = i;
                }
            }

            if (split_cfg->split_num == 0) {     //   说明本个模型的 Split 算子的 split 参数是作为 input 传入的
                // first in_operand is ifmap; second is split
                std::string split_name = std::string(base_cfg->in_operand_name[1]);
                OPERAND_S* split_operand = init_info_map[split_name];
                int64_t * split_ptr = (int64_t*)((char*)split_operand + sizeof(OPERAND_S));

                split_cfg->split_num = operand_elem_size(split_operand);
                for (int i = 0; i < split_cfg->split_num; ++i) {
                    split_cfg->split[i] = split_ptr[i];
                }
                base_cfg->in_operand_num = 1;   //  split operand have been trans to attr
            }


        }
        else if (op_type == "TopK") {
            TOP_K_CONFIG_S *top_k_cfg = (TOP_K_CONFIG_S *) cur_node_cfg_ptr;

            // Concat other attributes
            auto attr = node.attribute();
            for (auto param: attr) {

                if (param.name() == "axis") {
                    top_k_cfg->axis = param.i();
                } else if (param.name() == "largest") {
                    top_k_cfg->largest = param.i();
                } else if (param.name() == "sorted") {
                    top_k_cfg->sorted = param.i();
                }
            }
            std::string topk_num_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S *topk_num_operand = init_info_map[topk_num_name];
            int64_t *topk_num_data_ptr = (int64_t *) ((char *) topk_num_operand + sizeof(OPERAND_S));
            top_k_cfg->topk_num = topk_num_data_ptr[0];
            top_k_cfg->op_base_cfg.in_operand_num = 1;
        }
        else if (op_type == "Squeeze")
        {
            SQUEEZE_CONFIG_S *squeeze_cfg = (SQUEEZE_CONFIG_S *)cur_node_cfg_ptr;

            for (int i = 0; i < SHAPE_LEN; ++i) {
                squeeze_cfg->axes[i] = -1;
            }
            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {

                if (param.name() == "axes")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        squeeze_cfg->axes[i++] = val;
                    }
                    squeeze_cfg->axes_num = i;
                }
            }

            if (squeeze_cfg->axes_num == 0) {
                std::string axes_name = std::string(base_cfg->in_operand_name[1]);
                OPERAND_S* axes_operand = init_info_map[axes_name];
                int64_t * axes_data_ptr = (int64_t*)((char*)axes_operand + sizeof(OPERAND_S));
                squeeze_cfg->axes_num = operand_elem_size(axes_operand);
                for (int i = 0; i < squeeze_cfg->axes_num; ++i) {
                    squeeze_cfg->axes[i] = axes_data_ptr[i];
                }
            }
        }
        else if (op_type == "Transpose")
        {
            TRANSPOSE_CONFIG_S *transpose_cfg = (TRANSPOSE_CONFIG_S *)cur_node_cfg_ptr;

            // transpose other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {
                if (param.name() == "perm")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        transpose_cfg->perm[i++] = val;
                    }
                    transpose_cfg->perm_num = i;
                }
            }
        }
        else if (op_type == "Pow")
        {
            POW_CONFIG_S *pow_cfg = (POW_CONFIG_S *)cur_node_cfg_ptr;
            base_cfg->in_operand_num = 1;   //  such as roi and scales trans to attr

            // first in_operand is ifmap; second is power num
            std::string power_num_name = std::string(base_cfg->in_operand_name[1]);
            OPERAND_S*  power_num_operand = init_info_map[power_num_name];
            float * pow_cfg_ptr = (float*)((char*)power_num_operand + sizeof(OPERAND_S));
            pow_cfg->power_num = pow_cfg_ptr[0];
        }
        else if (op_type == "Unsqueeze")
        {
            UNSQUEEZE_CONFIG_S *unsqueeze_cfg = (UNSQUEEZE_CONFIG_S *)cur_node_cfg_ptr;

            for (int i = 0; i < SHAPE_LEN; ++i) {
                unsqueeze_cfg->axes[i] = -1;
            }
            // Concat other attributes
            auto attr = node.attribute();
            for (auto param : attr)
            {

                if (param.name() == "axes")
                {
                    int i = 0;
                    for (auto val : param.ints())
                    {
                        unsqueeze_cfg->axes[i++] = val;
                    }
                    unsqueeze_cfg->axes_num = i;
                }
            }

            if (unsqueeze_cfg->axes_num == 0) {
                std::string axes_name = std::string(base_cfg->in_operand_name[1]);
                OPERAND_S* axes_operand = init_info_map[axes_name];
                int64_t * axes_data_ptr = (int64_t*)((char*)axes_operand + sizeof(OPERAND_S));
                unsqueeze_cfg->axes_num = operand_elem_size(axes_operand);
                for (int i = 0; i < unsqueeze_cfg->axes_num; ++i) {
                    unsqueeze_cfg->axes[i] = axes_data_ptr[i];
                }
            }
//            LOG_DBG("Unsqueeze axes is %d %d %d", unsqueeze_cfg->axes[0], unsqueeze_cfg->axes[1], unsqueeze_cfg->axes[2]);
        }

		// update cur_node_cfg_ptr
		cur_node_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type]);
	}
}

void fill_init_info(const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &nodes,
                    const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &init_info,
                    char *init_info_ptr)
{
	char *cur_init_info_ptr = init_info_ptr;

    // 步骤 1： 填充 constant 数据
    // 这里将 constant 算子的输出也作为 init 数据，为其准备空间
    for (auto node : nodes) {
        auto op_type = node.op_type();
        auto name = node.name();

        if (op_type != "Constant") {
            continue;
        }

        // 开始填充 constant 数据
        int32_t operand_idx = 0;
        for (auto attr: node.attribute()) {
            if (attr.name() != "value") {
                continue;
            }
            OPERAND_S *operand_ptr = (OPERAND_S *)cur_init_info_ptr;
            std::string cur_operand_name = node.output()[operand_idx++];
            strcpy(operand_ptr->operand_name, cur_operand_name.c_str());
            operand_ptr->is_fixed_val = TRUE;

            auto constant_tensor = attr.t();
            auto data_type = constant_tensor.data_type();
            operand_ptr->data_type = (ELEM_TYPE_E)data_type;

            auto dims = constant_tensor.dims();
            std::vector<int32_t> vec_dim;
            for (auto dim : dims)
            {
                vec_dim.push_back((int32_t)dim);
            }
            operand_ptr->dim_num_of_shapes = vec_dim.size();
            for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
                operand_ptr->shapes[dim_i] = 1;
            }

            for (int i = 0; i < vec_dim.size(); ++i) {
                operand_ptr->shapes[i] = vec_dim[i];
            }

            auto raw_data = constant_tensor.raw_data();
            char *data_ptr = (char *)raw_data.c_str(); // read raw_data
            char *init_data_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
            int init_size = operand_buf_size(operand_ptr);
            memcpy(init_data_ptr, data_ptr, init_size * sizeof(int8_t));

            // update cur_init_info_ptr
            cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
        }
    }

    // 步骤 2： 填充 init 数据
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

        operand_ptr->dim_num_of_shapes = vec_dim.size();
        for (int dim_i = 0; dim_i < SHAPE_LEN; ++dim_i) {
            operand_ptr->shapes[dim_i] = 1;
        }
        if (vec_dim.size() == 0)
        {
            operand_ptr->dim_num_of_shapes = 1; // only one scalar
        }

        for (int i = 0; i < vec_dim.size(); ++i) {
            operand_ptr->shapes[i] = vec_dim[i];
        }
		auto raw_data = init_.raw_data();
		char *data_ptr = (char *)raw_data.c_str(); // read raw_data
		char *init_data_ptr = (char *)(cur_init_info_ptr + sizeof(OPERAND_S));
		int init_size = operand_buf_size(operand_ptr);
		memcpy(init_data_ptr, data_ptr, init_size * sizeof(int8_t));

        // 有的特殊的 init 数据，是放在 int64_data 而不是 raw_data 中
        if (raw_data.size() == 0) {
            auto tmp_raw_data = init_.int64_data();
            int elem_size = operand_elem_size(operand_ptr);
            int64_t *init_data_s64_ptr = (int64_t *)init_data_ptr;
            for (int i = 0; i < elem_size; ++i) {
                init_data_s64_ptr[i] = tmp_raw_data[i];
            }
        }

		// update cur_init_info_ptr
		cur_init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
	}
}

int insert_unique(std::vector<NODE_INFO_S>& vec, NODE_INFO_S& value) {
    // 检查元素是否已存在
    for (auto i : vec) {
        if (value.op_type == i.op_type && value.op_name == i.op_name) {
            return 0;
        }
    }
    // 若元素不存在，则插入新元素
    vec.push_back(value);
}

int fill_producer_and_consumer(char* one_file_buf) {

    Manager &m = Manager::getInstance();

    ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)one_file_buf;
    char* node_cfg_ptr = (char*)((char*)one_model_info_ptr + one_model_info_ptr->node_cfg_offset);
    char* init_info_ptr = (char*)((char*)one_model_info_ptr + one_model_info_ptr->init_info_offset);
    char* io_cfg_ptr = (char*)((char*)one_model_info_ptr + one_model_info_ptr->io_cfg_offset);
    int32_t node_cnt = one_model_info_ptr->node_cnt;
    int32_t init_cnt = one_model_info_ptr->init_cnt;
    int32_t io_cnt = one_model_info_ptr->io_cfg_cnt;

    std::vector<BASE_CONFIG_S*> base_op_vec;

    // 步骤 0： 直接把把所有 op 的 cfg 的地址全部放到 vector<base_op_cfg*> 中
    for (int node_i = 0; node_i < node_cnt; ++node_i) {
        BASE_CONFIG_S* base_op_ptr = (BASE_CONFIG_S*)node_cfg_ptr;
        base_op_vec.push_back(base_op_ptr);
        // update cur_node_cfg_ptr
        node_cfg_ptr += align_buf_size(m.op_cfg_size_map[base_op_ptr->op_type]);
    }

    // 步骤 1： 建立 operands_producer_map 和 operands_consumer_map 两个空的 map,
    // 遍历所有的 node_cfg_ptr，以该 op 的 in operands 为键，以 op name 值，往对应的 operands_consumer_map 中填充 op;
    // 再以该 op 的 out operands 为键，以 op name 值，往对应的 operands_producer_map 中填充 op;
    std::unordered_map<std::string, std::vector<NODE_INFO_S> > operands_producer_map;
    std::unordered_map<std::string, std::vector<NODE_INFO_S> > operands_consumer_map;
    for (int node_i = 0; node_i < node_cnt; ++node_i) {
        BASE_CONFIG_S* cur_op_ptr = (BASE_CONFIG_S*)base_op_vec[node_i];
        NODE_INFO_S cur_op_type_name;
        memcpy(&cur_op_type_name, cur_op_ptr, sizeof(NODE_INFO_S));
        for (int in_operand_i = 0; in_operand_i < cur_op_ptr->in_operand_num ; ++in_operand_i) {
            std::string key = cur_op_ptr->in_operand_name[in_operand_i];
            NODE_INFO_S val = cur_op_type_name;
            insert_unique(operands_consumer_map[key], val);
        }

        for (int out_operand_i = 0; out_operand_i < cur_op_ptr->out_operand_num ; ++out_operand_i) {
            std::string key = cur_op_ptr->out_operand_name[out_operand_i];
            NODE_INFO_S val = cur_op_type_name;
            insert_unique(operands_producer_map[key], val);
        }
    }

    // 把 io operand 加入到 operands_producer_map 和 operands_consumer_map 中
    for (int io_i = 0; io_i < io_cnt; io_i++) {
        IO_CONFIG_S* cur_io_ptr = (IO_CONFIG_S*)io_cfg_ptr;
        NODE_INFO_S cur_io_type_name;

        memcpy(&cur_io_type_name, cur_io_ptr, OP_TYPE_LEN);
        memcpy((char*)&cur_io_type_name + OP_TYPE_LEN, cur_io_ptr->operand.operand_name, OP_NAME_LEN);

        std::string key = cur_io_ptr->operand.operand_name;
        NODE_INFO_S val = cur_io_type_name;

        if (strcmp(cur_io_ptr->op_name, "input") == 0) {
            insert_unique(operands_producer_map[key], val);
        } else if (strcmp(cur_io_ptr->op_name, "output") == 0) {
            insert_unique(operands_consumer_map[key], val);
        } else {
            LOG_ERR("the io type should be input or output\n");
        }

        // update cur_io_cfg_ptr
        std::string op_type_str(io_cfg_ptr);
        io_cfg_ptr += align_buf_size(m.op_cfg_size_map[op_type_str]);
    }

    // 步骤 2： 遍历所有的 init_info_ptr，拿到所有的 init operands 存放到 set 中
    std::unordered_set<std::string> init_operands_set;
    for (int init_i = 0; init_i < init_cnt; ++init_i) {
        OPERAND_S *operand_ptr = (OPERAND_S *)init_info_ptr;
        init_operands_set.insert(init_info_ptr);

        // update init_info_ptr
        int init_size = operand_buf_size(operand_ptr);
        init_info_ptr += align_buf_size(sizeof(OPERAND_S) + init_size);
    }

    // 这些 op 就是本个 op 的 producer，把这些 op 填充到 base_op_cfg 的 producer 中; 同理用同样的方法填充 consumer
    for (int node_i = 0; node_i < node_cnt; ++node_i) {
        BASE_CONFIG_S* cur_op_ptr = (BASE_CONFIG_S*)base_op_vec[node_i];

        std::vector<NODE_INFO_S> cur_op_producer_vec;
        std::vector<NODE_INFO_S> cur_op_consumer_vec;
        // 遍历本个 op 的输入操作数，并且到 operands_producer_map 中去取这个输入操作数的 node，写到本个 op 的 cur_op_producer_vec 中
        for (int in_operand_i = 0; in_operand_i < cur_op_ptr->in_operand_num ; ++in_operand_i) {
            char* cur_in_operand = cur_op_ptr->in_operand_name[in_operand_i];

            if (init_operands_set.find(cur_in_operand) == init_operands_set.end()) {
                auto in_operand_vec = operands_producer_map[cur_in_operand];
                int32_t producer_num = operands_producer_map[cur_in_operand].size();
                memcpy(cur_op_ptr->producer[cur_op_ptr->producer_num].op_type, in_operand_vec[0].op_type, producer_num * sizeof(NODE_INFO_S));
                cur_op_ptr->producer_num += producer_num;
            }
        }

        // 遍历本个 op 的输出操作数，并且到 operands_consumer_map 中去取这个输出操作数的 node，写到本个 op 的 cur_op_consumer_vec 中
        for (int out_operand_i = 0; out_operand_i < cur_op_ptr->out_operand_num ; ++out_operand_i) {
            char* cur_out_operand = cur_op_ptr->out_operand_name[out_operand_i];

            if (init_operands_set.find(cur_out_operand) == init_operands_set.end()) {
                auto out_operand_vec = operands_consumer_map[cur_out_operand];
                int32_t consumer_num = operands_consumer_map[cur_out_operand].size();
                memcpy(cur_op_ptr->consumer[cur_op_ptr->consumer_num].op_type, out_operand_vec[0].op_type, consumer_num * sizeof(NODE_INFO_S));
                cur_op_ptr->consumer_num += consumer_num;
            }
        }
    }
    return 0;
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
	if (argc != 2 && argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " [onnx_path] [one_path]" << std::endl;
		std::cout << "Usage: " << argv[0] << " [yml_path]" << std::endl;
		exit(-1);
	}

    std::string onnx_path_str,  one_path_str;
    if (argc == 2) {
        const char *rt_cfg_txt = argv[1];
        std::string rt_cfg_txt_str(rt_cfg_txt);
        std::unordered_map<std::string, std::string> cfg_info_map;
        yml2map(cfg_info_map, rt_cfg_txt_str);
        onnx_path_str = cfg_info_map["onnx_file_path"];
        one_path_str = cfg_info_map["one_file_path"];
    } else {
        onnx_path_str = argv[1];
        one_path_str = argv[2];
    }

    const char *onnx_path = onnx_path_str.c_str();
    const char *one_path = one_path_str.c_str();

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

	// step 3: get node size of this onnx model, and traverse each node to get op cfg size
	int32_t node_cnt, align_cfg_size;
	if (get_align_cfg_size(graph.node(), node_cnt, align_cfg_size) != 0)
	{
        LOG_ERR("failed: some op not being implemented temporarily!");
		return -1;
	}

	// step 3: traverse initializer to get init size (including the values of weights and bias)
	int32_t init_cnt, align_init_size;
	get_align_init_size(graph.node(), graph.initializer(), init_cnt, align_init_size);

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
        LOG_ERR("failed: malloc for one file");
		return -1;
	}

	// step 5: fill the head information
    ONE_MODEL_DESC_S *one_model_info_ptr = (ONE_MODEL_DESC_S *)one_file_buf;
    one_model_info_ptr->node_cnt = node_cnt;
    one_model_info_ptr->node_cfg_offset = head_size; // the offset of first node cfg from one_file_buf
    one_model_info_ptr->init_cnt= init_cnt;
    one_model_info_ptr->init_info_offset = head_size + align_cfg_size; // the offset of first init info from one_file_buf
    one_model_info_ptr->io_cfg_cnt = io_cfg_cnt;
    one_model_info_ptr->io_cfg_offset = head_size + align_cfg_size + align_init_size; // the offset of first io op cfg from one_file_buf

    // step 6: fill the init info
    char *init_info_ptr = (char *)(one_file_buf + one_model_info_ptr->init_info_offset);
    fill_init_info(graph.node(), graph.initializer(), init_info_ptr);

	// step 7: fill the node cfg
	char *node_cfg_ptr = (char *)(one_file_buf + one_model_info_ptr->node_cfg_offset);
	fill_node_cfg(graph.node(), node_cfg_ptr, init_cnt, init_info_ptr);

	// step 8: fill the io cfg
	char *io_cfg_ptr = (char *)(one_file_buf + one_model_info_ptr->io_cfg_offset);
	fill_io_cfg(graph.input(), graph.output(), io_cfg_ptr);

    // step 9: fill producer and consumer of each op
    fill_producer_and_consumer(one_file_buf);

    // step 10: dump .one file
	FILE *file_p = fopen(one_path, "w");
	fwrite((void *)one_file_buf, 1, one_file_size, file_p);
	fclose(file_p);

    free(one_file_buf);
    onnx_file.close();
    return 0;
}

