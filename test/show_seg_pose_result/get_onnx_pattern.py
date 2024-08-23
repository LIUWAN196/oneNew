import onnx

input_path = "/home/wanzai/桌面/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/todo/swin_t.onnx"
output_path = "/home/wanzai/桌面/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/todo/swin_t_st2.onnx"
input_names = ["input0"]
output_names = ["/layers/layers.0/blocks/blocks.0/Add_3_output_0"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)

