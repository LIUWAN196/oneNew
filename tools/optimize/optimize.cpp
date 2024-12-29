#include "optimize.hpp"


int32_t opt_cfg_check(std::unordered_map<std::string, std::string>& cfg_info_map){

    std::string optimize_type = str2lower_str(cfg_info_map["optimize_type"]);
    if (optimize_type.empty()) {
        LOG_ERR("the args: optimize_type must be set, for example: op_fusion or op_fusion&model_quant");
        return -1;
    }

    std::string one_file_path = cfg_info_map["one_file_path"];
    if (one_file_path.empty()) {    // default args: false
        LOG_ERR("the args: one_file_path must be set");
        return -1;
    }

    std::string opt_one_file_path = cfg_info_map["opt_one_file_path"];
    if (opt_one_file_path.empty()) {
        set_default_args(cfg_info_map, "opt_one_file_path", one_file_path);
    }

    std::string do_quant = "model_quant";
    size_t pos = optimize_type.find(do_quant);

    if (pos == std::string::npos) {
        // don't need quant from the model_quant info
        return 0;
    }

    // into there, the optimize_type include model_quant
    std::string calibrate_img_name_txt_path = cfg_info_map["calibrate_img_name_txt_path"];
    if (calibrate_img_name_txt_path.empty()) {
        LOG_ERR("the args: calibrate_img_name_txt_path must be set");
        return -1;
    }

    std::string calibrate_img_folder = cfg_info_map["calibrate_img_folder"];
    if (calibrate_img_folder.empty()) {
        LOG_ERR("the args: calibrate_img_folder must be set");
        return -1;
    }

    std::string calibrate_img_num = cfg_info_map["calibrate_img_num"];
    if (calibrate_img_num.empty()) {
        LOG_ERR("the args: calibrate_img_num must be set");
        return -1;
    }

    std::string quant_type = str2lower_str(cfg_info_map["quant_type"]);
    if (quant_type.empty()) {    // default args: true
        set_default_args(cfg_info_map, "quant_type", "kl");
    }
    if (str2lower_str(cfg_info_map["quant_type"]) != "kl" &&
        str2lower_str(cfg_info_map["quant_type"]) != "mse" &&
        str2lower_str(cfg_info_map["quant_type"]) != "percent") {
        LOG_ERR("the args: quant_type must be set: kl、mse、percent");
        return -1;
    }

    std::string resize_shapes = cfg_info_map["resize_shapes"];
    if (resize_shapes.empty()) {
        LOG_ERR("the args: resize_shapes must be set, for example: [256, 256]");
        return -1;
    }

    std::string crop_shapes = cfg_info_map["crop_shapes"];
    if (crop_shapes.empty()) {
        LOG_ERR("the args: crop_shapes must be set, for example: [224, 224]");
        return -1;
    }

    std::string normal_mean = cfg_info_map["normal_mean"];
    if (normal_mean.empty()) {
        LOG_ERR("the args: normal_mean must be set, for example: [0.485ff, 0.456ff, 0.406ff]");
        return -1;
    }

    std::string normal_std = cfg_info_map["normal_std"];
    if (normal_mean.empty()) {
        LOG_ERR("the args: normal_std must be set, for example: [0.229f, 0.224f, 0.225f]");
        return -1;
    }

    return 0;
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        LOG_ERR("Usage: %s [optimize config.yml]，for example: model_and_cfg_zoo/configs/samples/optimize_sample.yml]", argv[0]);
    }

    const char *rt_cfg_txt = argv[1];
    std::string rt_cfg_txt_str(rt_cfg_txt);
    std::unordered_map<std::string, std::string> cfg_info_map;
    yml2map(cfg_info_map, rt_cfg_txt_str);

    opt_cfg_check(cfg_info_map);

    const char* one_file_path = cfg_info_map["one_file_path"].c_str();
    const char* opt_one_file_path = cfg_info_map["opt_one_file_path"].c_str();

    // step 1: get one file size
    std::ifstream one_file(one_file_path, std::ios::ate | std::ios::binary);
    int32_t one_file_size = static_cast<int32_t>(one_file.tellg());
    cfg_info_map["one_file_size"] = std::to_string(one_file_size);
    one_file.close();

    // step 2: load one file
    char *one_buf_ptr = (char *)aligned_alloc(32, one_file_size);
    FILE *file_p = NULL;

    file_p = fopen(one_file_path, "r");
    if (file_p == NULL)
    {
        LOG_ERR("failed: can't open the one file");
        return 0;
    }
    size_t ret = fread(one_buf_ptr, sizeof(char), one_file_size, file_p);
    fclose(file_p);

    std::string optimize_type = cfg_info_map["optimize_type"];

    // do op fusion
    size_t do_fusion = optimize_type.find("op_fusion");
    if (do_fusion != std::string::npos) {
        char *fused_one_buf_ptr = (char *)aligned_alloc(32, one_file_size);
        op_fusion(fused_one_buf_ptr, one_buf_ptr, cfg_info_map);

        char* tmp_ptr = one_buf_ptr;
        one_buf_ptr = fused_one_buf_ptr;
        free(tmp_ptr);
    }

    // do model quant
    size_t do_quant = optimize_type.find("model_quant");
    if (do_quant != std::string::npos) {
        char *quant_one_buf_ptr = (char *)aligned_alloc(32, one_file_size);
        model_quant(quant_one_buf_ptr, one_buf_ptr, cfg_info_map);

        char* tmp_ptr = one_buf_ptr;
        one_buf_ptr = quant_one_buf_ptr;
        free(tmp_ptr);
    }

    // step 10: dump the optimize_one_buf_ptr as .one
    FILE *opt_file_p = fopen(opt_one_file_path, "w");
    fwrite((void *)one_buf_ptr, 1, one_file_size, opt_file_p);
    fclose(opt_file_p);

    free(one_buf_ptr);
    return 0;
}

