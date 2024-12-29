int do_pad_conv(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *cfg) {
    float pad_value = 0;
    float *dst_f32 = (float *) dst_ptr;
    float *src_f32 = (float *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + 2 * cfg->h;
    int32_t dst_w = src_w + 2 * cfg->w;

    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = cfg->h; h_i < dst_h - cfg->h; ++h_i) {
            for (int w_i = cfg->w; w_i < dst_w - cfg->w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] = src_f32[c_i * src_h * src_w +
                                                                           (h_i - cfg->h) * src_w + (w_i - cfg->w)];
            }

        }
    }

    return 0;
}

int do_pad_conv_s8(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *cfg) {
    int8_t pad_value = 0;
    int8_t *dst_f32 = (int8_t *) dst_ptr;
    int8_t *src_f32 = (int8_t *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + 2 * cfg->h;
    int32_t dst_w = src_w + 2 * cfg->w;

    for (int i = 0; i < dst_c * dst_h * dst_w; ++i) {
        dst_f32[i] = pad_value;
    }

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        for (int h_i = cfg->h; h_i < dst_h - cfg->h; ++h_i) {
            for (int w_i = cfg->w; w_i < dst_w - cfg->w; ++w_i) {
                dst_f32[c_i * dst_h * dst_w + h_i * dst_w + w_i] = src_f32[c_i * src_h * src_w +
                                                                           (h_i - cfg->h) * src_w + (w_i - cfg->w)];
            }

        }
    }

    return 0;
}

int do_pad_conv_new(char *dst_ptr, char *src_ptr, OPERAND_S *src_data_desc, PAD_INNER_CONFIG_S *pad_cfg) {

    float *dst_f32 = (float *) dst_ptr;
    float *src_f32 = (float *) src_ptr;

    int32_t src_c = src_data_desc->shapes[1];
    int32_t src_h = src_data_desc->shapes[2];
    int32_t src_w = src_data_desc->shapes[3];

    int32_t dst_c = src_data_desc->shapes[1];
    int32_t dst_h = src_h + pad_cfg->top_pad + pad_cfg->bottom_pad;
    int32_t dst_w = src_w + pad_cfg->left_pad + pad_cfg->right_pad;

    memset(dst_ptr, 0, dst_c * dst_h * dst_w * sizeof(float));

    const int32_t top_pad = pad_cfg->top_pad;
    const int32_t bottom_pad = pad_cfg->bottom_pad;
    const int32_t left_pad = pad_cfg->left_pad;

    for (int c_i = 0; c_i < dst_c; ++c_i) {
        float *cur_src_f32 = src_f32 + c_i * src_h * src_w - top_pad * src_w;
        float *cur_dst_f32 = dst_f32 + c_i * dst_h * dst_w + left_pad;
        int h_i = top_pad;
        for (h_i = top_pad; h_i < dst_h - bottom_pad - 3; h_i += 4) {
            memcpy(cur_dst_f32 + (h_i + 0) * dst_w, cur_src_f32 + (h_i + 0) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 1) * dst_w, cur_src_f32 + (h_i + 1) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 2) * dst_w, cur_src_f32 + (h_i + 2) * src_w, src_w * sizeof(float));
            memcpy(cur_dst_f32 + (h_i + 3) * dst_w, cur_src_f32 + (h_i + 3) * src_w, src_w * sizeof(float));
        }
        for (; h_i < dst_h - bottom_pad; h_i++) {
            memcpy(cur_dst_f32 + h_i * dst_w, cur_src_f32 + h_i * src_w, src_w * sizeof(float));
        }
    }

    return 0;
}
