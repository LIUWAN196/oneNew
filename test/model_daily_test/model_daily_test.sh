#!/bin/bash
# 默认情况下，如果一个命令失败了（返回非零退出状态），shell 脚本会继续执行下一个命令。但是，使用 set -e，一旦遇到错误，脚本就会停止执行。这里的 e 是 end 的意思
set -e

onnx_dir="/home/wanzai/桌面/model_and_cfg_and_ofmap_folder/model_and_cfg_zoo/model/onnx_model"
sh_script_dir=$(dirname $(readlink -f "$0"))
model_yml_dir=${sh_script_dir}/model_daily_test_yml
root_dir=$(dirname $(dirname "$sh_script_dir"))

model_yml_list=""

#echo "sh_script_dir is ${sh_script_dir}, root_dir is ${root_dir}"

function get_model_list() {
    # 使用for循环遍历目录中的文件
    for model_cfg in "$model_yml_dir"/*; do
        if [ -f "$model_cfg" ]; then
            model_yml_list="$model_yml_list $model_cfg"
        fi
    done
}

function model_daily_test_main() {

    if [ $# -eq 1 ];then
        # 只测试指定的单个模型，仅仅支持绝对路径。
        # 例如：bash ./test/model_daily_test/model_daily_test.sh /home/wanzai/桌面/oneNew/test/model_daily_test/model_daily_test_yml/yolov10s.yml
        model_yml_list="$model_yml_list $1"
    else
        # 测试所有模型
        # step 0: get the list of models to be tested
        get_model_list
    fi

    # step 1: traverse each model and execute the testing process
    model_cnt=0
    for model_cfg in $model_yml_list; do
        # start test model
        model_name=$(basename ${model_cfg} .yml)
        # echo "model name is ${model_name}"

        # step 1.1: prepare tmp_one_model_path and tmp_ofmap_path
        if [ -d ${sh_script_dir}/tmp_one_model_path/${model_name} ]; then
            rm -rf ${sh_script_dir}/tmp_one_model_path/${model_name}
        fi
        mkdir -p ${sh_script_dir}/tmp_one_model_path/${model_name}

        if [ -d ${sh_script_dir}/tmp_ofmap_path/${model_name} ]; then
            rm -rf ${sh_script_dir}/tmp_ofmap_path/${model_name}
        fi
        mkdir -p ${sh_script_dir}/tmp_ofmap_path/${model_name}

        # step 1.2: do onnx2one
        cd ${root_dir}/cmake-build-debug/tools/onnx2one
        ./onnx2one ${model_cfg}

        # step 1.3: do model_infer
        cd ${root_dir}/cmake-build-debug/example
        ./model_infer ${model_cfg}

        # step 1.4: using onnx runtime to compare result
        cd ${sh_script_dir}
        python ./compute_ofmap_consim.py ${onnx_dir}/${model_name}".onnx" ${sh_script_dir}/tmp_ofmap_path/${model_name}/

        let model_cnt=model_cnt+1
    done
    echo -e "\033[32m        ********  end normal model daily test, total model cnt is: ${model_cnt}  ********        \033[0m"

    # step 2: start test opt model
    model_cnt=0
    for model_cfg in $model_yml_list; do
        # step 2.1: do model opt
        cd ${root_dir}/cmake-build-debug/tools/optimize
        ./optimize ${model_cfg}

        # step 2.2: do model_infer
        cd ${root_dir}/cmake-build-debug/example
        ./model_infer ${model_cfg}

        # step 2.3: using onnx runtime to compare result
        cd ${sh_script_dir}
        python ./compute_ofmap_consim.py ${onnx_dir}/${model_name}".onnx" ${sh_script_dir}/tmp_ofmap_path/${model_name}/

        let model_cnt=model_cnt+1
    done
    echo -e "\033[32m        ********  end optimize model daily test, total model cnt is: ${model_cnt}  ********        \033[0m"
}

# 上面的所有内容都是 function 修改了的。所以会一路向下寻找，找到这个 main。这个 main 实际上就是一个平常的函数，调用上面的 function main() 函数
# 这里的 $@ 是获取所有向脚本传递的参数
#main $@

model_daily_test_main $@









