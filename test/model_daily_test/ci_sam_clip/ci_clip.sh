#!/bin/bash
# 默认情况下，如果一个命令失败了（返回非零退出状态），shell 脚本会继续执行下一个命令。但是，使用 set -e，一旦遇到错误，脚本就会停止执行。这里的 e 是 end 的意思
set -e

onnx_dir="/home/wanzai/桌面/oneNew/model_zoo/onnx_model"
root_dir="/home/wanzai/桌面/oneNew"

function ci_clip_main() {
    echo -e "\033[32m        ********  test clip model  ********        \033[0m"
    # do decoder onnx2one
    if [ -d ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_img ]; then
        rm -rf ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_img
    fi
    mkdir -p ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_img

    cd ${root_dir}/build/tools/onnx2one
    ./onnx2one ${onnx_dir}/clip_img.onnx ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_img/clip_img.one

    # do encoder onnx2one
    if [ -d ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_txt ]; then
        rm -rf ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_txt
    fi
    mkdir -p ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_txt

    cd ${root_dir}/build/tools/onnx2one
    ./onnx2one ${onnx_dir}/clip_txt.onnx ${root_dir}/test/model_daily_test/tmp_one_model_path/clip_txt/clip_txt.one

    cd ${root_dir}/build/example
    ./model_infer ${root_dir}/test/model_daily_test/ci_sam_clip/clip.yml
}

# 上面的所有内容都是 function 修改了的。所以会一路向下寻找，找到这个 main。这个 main 实际上就是一个平常的函数，调用上面的 function main() 函数
# 这里的 $@ 是获取所有向脚本传递的参数
#main $@

ci_clip_main $@









