#!/bin/bash
# 默认情况下，如果一个命令失败了（返回非零退出状态），shell 脚本会继续执行下一个命令。但是，使用 set -e，一旦遇到错误，脚本就会停止执行。这里的 e 是 end 的意思
set -e

# 下面这句话的意思如下：提取 .sh 所在的绝对路径并赋值给 root_dir
# readlink -f "$0" 是一个在Shell脚本中常用的命令，用于获取脚本自身的绝对路径。
# readlink 是一个命令，用于显示一个符号链接指向的目标。  -f 是一个选项，用于获取链接目标的绝对路径。  "$0" 是一个特殊的Shell变量，代表脚本自身的名称。在执行脚本时，$0会被替换为脚本的路径。
# dirname 命令会返回路径中的目录部分
root_dir=$(dirname $(readlink -f "$0"))

# 根据 root_dir 这个绝对路径，将其中一个下级目录赋值给 sdk_install_dir。(这个时候这个路径是否存在都无所谓，此时只是一个字符串而已，在后面使用时 mkdir 这个目录即可)
sdk_install_dir=${root_dir}/build/install/x86
test_op_name_list=""

# 声明两个变量
test_exe=es_dsp_execution
compiler_exe=es_dsp_desc

# 使用带颜色的字体打印。下面的 NC 是结束带颜色字体打印的意思。    例如    echo -e "${RED}There is no compare log file for case:$yml_name.${NC}"
# 上面个这个 -e 表示：在打印过程中，需要把 \ 当作转义字符看待，而不是单做普通的 \ 看待。例如：echo "Hello\nWorld" 打印出来是 Hello\nWorld。但是如果使用 echo -e "Hello\nWorld"
# 输出就是
# Hello
# World
RED='\e[1;31m'
GREEN='\e[1;32m'
YELLOW='\e[1;33m'
NC='\e[0m'

# 这个函数是打印有效的帮助信息，帮助使用者使用本个 .sh。其中 $0 表示的就是本个 .sh 的名字； -e 就是让 \ 被视作转义字符
function usage() {
    echo -e "Usage:\n$0 <test_yml_file | test_yml_dir> [win2030_root_dir]"
    echo -e "e.g.:\n$0 es_vecadd.yml [win2030_root_dir]"
    echo -e "$0 eswin/tests/py_scripts/sample_yml [win2030_root_dir]"
}

# 做编译前的准备
function pre_build() {
# 在shell脚本中，local 用于声明局部变量。当我们在函数中定义变量时，可以使用local关键字来限定变量的作用域，使其只在函数内部有效，避免变量名冲突和不必要的变量污染。

    local str_1=$1
    # -d 检查 ${test_dir} 这个目录是否存在；类似的， -e 检查指定的文件是否存在
    # 下面这个判断语句中多了一个 ！。所以含义是：如果不存在 ${test_dir}，则新建这样一个文件夹
    if [ ! -d ${test_dir} ]; then
    # 这里的 -p 含义是如果目标目录的上级目录不存在，-p 选项会递归地创建所有不存在的父目录，无需手动一级一级创建
        mkdir -p ${test_dir}
    fi

# 如果有 ${test_op_case_dir}，则将这个目录删除并 mkdir 这个目录
    if [ -d ${test_op_case_dir} ]; then
        rm -rf ${test_op_case_dir}
    fi
    mkdir -p ${test_op_case_dir}
}


function prepare_op_cases() {
    # 读取传入本函数的第一个参数
    input_yml=$1

    #设置本次计算的环境
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${sdk_install_dir}/host/x86_64/lib
#===========================================================
    # -f ${input_yml} ： 看这个路径是不是一个文件
    if [ -f ${input_yml} ]; then
    # 如果是一个文件，提取后缀为 extension，下面这个 if 语句是看后缀是否为 yml
        extension="${input_yml##*.}"
        if [[ $extension != "yml" ]]; then
            echo "${input_yml} is not the yml file."
            usage
            exit 1
        fi

# 从${input_yml}变量所表示的路径中提取基本文件名，并删除.yml扩展名。返回给 yml_name
        yml_name=$(basename ${input_yml} .yml)
        mkdir -p ${test_op_case_dir}/${yml_name}
        # 复制 yml 到待测试的目录
        cp ${input_yml} ${test_op_case_dir}/${yml_name}
    fi

    # 获取待测试的 yml 列表
    tmp_input_list=$(find ${test_op_case_dir} -name "*.yml")
    if [[ ${tmp_input_list} == "" ]]; then
        echo "there is no yml in ${input_yml}"
        exit 1
    fi

# 使用 for 循环遍历待测试的 yml 列表
    for yml_file in $tmp_input_list
    do
        echo "test yml_file =${yml_file}"
    done

}

function compare_dsp_output() {
    passed_tests=0

    yml_list=$(find ${test_op_case_dir} -name "*.yml")
    total_cases=$(echo "$yml_list" | wc -l)

    for yml_file in ${yml_list}
    do
        yml_name=$(basename ${yml_file} .yml)
        yml_dir=$(dirname ${yml_file})
        log_file=${yml_dir}/compare.log
        # 执行 python
        python ${py_data_tool} ${yml_file} ${yml_dir}/${yml_name} "generate"
        python ${py_data_tool} ${yml_file} ${yml_dir} "compare" > ${log_file}       # 将 compare 的结果重定向到 log_file 里面，而不是直接打印出来

    done

}

function main() {
    # shell 中的 if 语句写法。其中 $# 表示传递到脚本的参数个数； -le 表示 <=，同理下面的 -eq 表示 =，
    if [ $# -le 0 ];then
    # 这里的 usage 是上面定义的函数，在这个函数中打印代码的使用方法和参数说明，它提供了一种参考或指导，帮助用户正确地使用代码或程序
        usage
        # 由于用户调用本个 sh 时，没有传入必要的参数，所以退出
        exit 1
    fi

    if [ $# -eq 2 ];then
    # 和其他编程语言一样。传入 shell 的 $0 为本个执行的 .sh 名称；后面 $1、 $2、 $3 是依次传入的参数
        win2030_root=$2
    fi

    echo "***************** setup test environment *****************"
    pre_build
    setup_py_env

    echo "***************** get test operator list *****************"
    prepare_op_cases $1

    echo "***************** generate operator test data *****************"
    gen_op_test_data ${test_op_case_dir}

    echo "***************** run dsp kernel test *****************"
    run_kernel_test

    echo -e "\n***************** compare result *****************"
    compare_dsp_output
}

# 这是我拿来练习的一个函数
function liu_test(){
# 在 shell 脚本中，echo 可以等价于 print
echo "hello, this is first shell"

# 在 shell 脚本中，变量名和等号之间不能有空格
v1="this is v1"
v2="this is v2"
num1="32"
num2="1232"

# 想使用之前定义过的变量，前面加上 $ 即可。这下面输出的是  "this is v1"   v2    (这里的 v2 因为没加 $ 所以被识别为字符 v2)
echo $v1,    v2
}

function liu_fun_0() {
  # print the roo_dir with red color
  echo -e "${RED} this root dir is: ${root_dir} ${NC}"

  # using usage function to 打印有效的帮助信息
  usage

  str_1="hello, shell"
  echo "the string is ${str_1}"

  dir_1="/home/e0006809/Desktop/test_sh_and_cmake"
  dir_2="/home/e0006809/Desktop/test_sh_and_cmake_new"
  # -d 检查 ${test_dir} 这个目录是否存在；类似的， -e 检查指定的文件是否存在
  # 下面这个判断语句中多了一个 ！。所以含义是：如果不存在 ${test_dir}，则新建这样一个文件夹
  if [ -d ${dir_1} ]; then
    echo "${dir_1} is exist"
  else
    # 这里的 -p 含义是如果目标目录的上级目录不存在，-p 选项会递归地创建所有不存在的父目录，无需手动一级一级创建
    mkdir -p ${dir_1}
  fi

  if [ -d ${dir_2} ]; then
    echo "${dir_2} is exist"
  else
    echo "${dir_2} is not exist, to make it"
    mkdir -p ${dir_2}
  fi

  # 如果有 ${dir_2}，则将这个目录删除并 mkdir 这个目录
  if [ -d ${dir_2} ]; then
    echo "remove ${dir_2}"
      rm -rf ${dir_2}
  fi
  echo "mkdir ${dir_2} second"
  mkdir -p ${dir_2}
}

function liu_fun_1() {
    param_1=$1
    param_2=$2
    echo "show the params"
    echo "the input param 1 is: ${param_1}"
    echo "the input param 2 is: ${param_2}"

    #设置本次计算的环境
    export LD_LIBRARY_PATH=/host/x86_64/lib

    # -f ${param_1} ： 看 param_1 是不是一个文件
    if [ -f ${param_1} ]; then
      # 如果是一个文件，提取后缀为 suffix
      suffix="${param_1##*.}"

      # 这个 if 语句是看后缀是否为 yml
      if [[ ${suffix} != "yml" ]]; then
        echo "${param_1} is not the yml file."
      else
        echo "${param_1} is the yml file."

        # 从${input_yml}变量所表示的路径中提取基本文件名，并删除.yml扩展名。返回给 yml_name
        name_without_suffix=$(basename ${param_1} .yml)
        echo "the param_1 without suffix is: ${name_without_suffix}"

        # 根据 name_without_suffix 新建一个文件夹，并且复制 yml 到这个目录
        new_dir=${root_dir}/${name_without_suffix}
        echo "start to make ${new_dir}"
        mkdir -p ${new_dir}
        # copy ${param_1} to new_dir
        cp ${param_1} ${new_dir}
      fi
    fi

    # 获取本文件夹中的 yml 列表
    echo "show the new_dir, the new_dir is: ${new_dir}"
    yml_list=$(find ${root_dir} -name "*.yml")
    # 使用 for 循环遍历，打印 yml 列表
    for yml_file in ${yml_list}
    do
        echo "test yml_file = ${yml_file}"
    done


}

function liu_fun_2() {
    # shell 中的 if 语句写法。其中 $# 表示传递到脚本的参数个数； -le 表示 <=，同理 -eq 表示 =，
    if [ $# -le 0 ];then
    # 这里的 usage 是上面定义的函数，在这个函数中打印代码的使用方法和参数说明，它提供了一种参考或指导，帮助用户正确地使用代码或程序
        usage
        # 由于用户调用本个 sh 时，没有传入必要的参数，所以退出
        exit 1
    fi

    log_file=${root_dir}/compare.log
    if [ ! -f ${log_file} ]; then
      echo "the log: ${log_file} is not exit, start to make it"
      touch ${log_file}
    fi


    echo "execute python with branch_1 to generate input.bin"
    python py_test.py $2 $3 "branch_1"


    echo "execute python with branch_2 to compare the correct of C"
    # 将下面 branch_2 的 python 内部的打印内容重定向到 log_file 里面，而不是在终端里直接打印出来
    python py_test.py $2 $3 "branch_2" > ${log_file}

    echo "the sys.exit is $?"

}


function liu_main() {
    liu_fun_0
    liu_fun_1 $@
    liu_fun_2 $@
}

# 上面的所有内容都是 function 修改了的。所以会一路向下寻找，找到这个 main。这个 main 实际上就是一个平常的函数，调用上面的 function main() 函数
# 这里的 $@ 是获取所有向脚本传递的参数
#main $@

liu_main $@





