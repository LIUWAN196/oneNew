import json
import csv
from csv import reader

# 加载 csv 并去除表头
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        #读取表头X，Y
        headings = next(csv_reader)
        #文件指针下一至第一条真正数据
        for row in csv_reader:
            if not row:   #判定是否有空行，如有，则跳入到下一行
                continue
            dataset.append(row)
    return dataset

# 记录重要 op 的信息
def add_item(major_op_info, op_type, op_computation, op_exc_time):
    # 检查name是否已存在
    for item in major_op_info:
        if item['op_type'] == op_type:
            item['op_num_cnt'] += 1
            item['op_computation'] += op_computation
            item['op_exc_time'] += op_exc_time
            return
    # 如果不存在，添加新元素
    major_op_info.append({'op_type': op_type, 'op_num_cnt': 1,
                          'op_computation': op_computation, 'op_exc_time': op_exc_time})


if __name__ == '__main__':
    # step 1：加载 csv
    csv_file_name = 'timeline_info/resnet50.csv'
    json_file_name = 'timeline_info/resnet50.json'

    dataset=load_csv(csv_file_name)

    hardware_and_model_info = dataset[0][6::]
    hardware_computing_power = float(hardware_and_model_info[4].strip())
    computing_power = hardware_computing_power * (1024 ** 3)

    # step 2：遍历 csv，将信息保存到 traceEvents 和 major_op_info 中
    timeline_json = {}
    timeline_data = json.loads(json.dumps(timeline_json))
    timeline_data['traceEvents'] = 'NONE'
    traceEvents = []
    major_op_info = []
    model_computation = 0
    for op_idx in range(len(dataset)):
        cur_op_info = dataset[op_idx]
        st_time_stamp = float(cur_op_info[2].strip())
        ed_time_stamp = float(cur_op_info[3].strip())
        computation = float(cur_op_info[4].strip()) * 1e6
        model_computation += computation
        op_exc_time = ed_time_stamp - st_time_stamp  # the unit is subtle
        efficiency = computation / (op_exc_time * 1e-6 * computing_power)

        exc_info = {'name': cur_op_info[0], 'ph': 'X', "pid": "exc_timeline",
                    "tid": cur_op_info[1], "ts": st_time_stamp, "dur": op_exc_time}
        st_efficiency_info = {'name': cur_op_info[1], 'ph': 'C', "pid": "model_efficiency",
                              "tid": cur_op_info[0], "ts": st_time_stamp, 'args': {'value': efficiency}}
        ed_efficiency_info = {'name': cur_op_info[1], 'ph': 'C', "pid": "model_efficiency",
                              "tid": cur_op_info[0], "ts": ed_time_stamp, 'args': {'value': -1}}

        traceEvents.append(exc_info)
        if (computation > 0):
            traceEvents.append(st_efficiency_info)
            traceEvents.append(ed_efficiency_info)
            add_item(major_op_info, cur_op_info[1], computation, op_exc_time)

    timeline_data['traceEvents'] = traceEvents

    # step 3：往 timeline_data 这个 json 中写入硬件信息和 model 信息
    timeline_data['======== HARDWARE INFO ========'] = "========================"
    timeline_data['cpu'] = 'null' if hardware_and_model_info[0] == "" else hardware_and_model_info[0]
    timeline_data['cpu_hardware_info'] = 'null' if hardware_and_model_info[1] == "" else hardware_and_model_info[1]
    timeline_data['gpu'] = 'null' if hardware_and_model_info[2] == "" else hardware_and_model_info[2]
    timeline_data['gpu_hardware_info'] = 'null' if hardware_and_model_info[3] == "" else hardware_and_model_info[3]
    hw_cpt_power = float(hardware_and_model_info[4].strip())
    timeline_data['total_hardware_computing_power (GOPS)'] = "{:.3f}".format(hw_cpt_power)

    model_eval_time_s = float(dataset[len(dataset) - 1][3].strip()) * 1e-6
    model_computation_gflops = model_computation / (1024 ** 3)
    timeline_data['======== MODEL INFO ========'] = "========================"
    timeline_data['model_name'] = hardware_and_model_info[5]
    timeline_data['op_num'] = len(dataset)
    timeline_data['model_computation (FLOPs(G))'] = "{:.3f}".format(model_computation_gflops)
    timeline_data['model_performance'] = "{:.3f}".format(1 / model_eval_time_s) + "fps"
    timeline_data['model_eval time'] = str("{:.3f}".format(model_eval_time_s * 1000)) + "ms"
    model_hw_efficiency = model_computation_gflops / (model_eval_time_s * hw_cpt_power) * 100
    timeline_data['model_hardware_efficiency'] = str("{:.3f}".format(model_hw_efficiency)) + "%"

    # step 4：对 major_op_info 按照 op_exc_time 降序排列，并往 timeline_data 这个 json 中写入重要 op 的信息
    major_op_info.sort(key=lambda x: x['op_exc_time'], reverse=True)
    for op_i in range(len(major_op_info)):
        op_type_upper = major_op_info[op_i]['op_type'].upper()
        op_type_lower = major_op_info[op_i]['op_type'].lower()
        op_num_cnt = major_op_info[op_i]['op_num_cnt']
        op_computation = major_op_info[op_i]['op_computation']
        op_exc_time = major_op_info[op_i]['op_exc_time']
        eval_time_ms = op_exc_time * 1e-3
        ratio_in_model_eval = "{:.3f}".format(eval_time_ms / (model_eval_time_s * 1e3) * 100)

        if (eval_time_ms / (model_eval_time_s * 1e3) <= 0.5 / 100):
            # 只有耗时占总模型比 > 0.5% 的，才认为是重要 op，才打印出这个类型 op 的信息，否则判断下一个类型 op
            continue
        timeline_data['======== ' + op_type_upper + ' INFO ========'] = "========================"
        timeline_data[op_type_lower + ' op_num_cnt'] = op_num_cnt
        timeline_data[op_type_lower + ' computation (FLOPs(G))'] = "{:.4f}".format(op_computation / (1024 ** 3))
        timeline_data[op_type_lower + ' eval time&ratio_in_model_eval'] = str("{:.3f}".format(eval_time_ms)) + "ms " \
                                                                          + str(ratio_in_model_eval) + "%"
        op_hw_efficiency = (op_computation / (1024 ** 3)) / (eval_time_ms / 1000 * hw_cpt_power) * 100
        timeline_data[op_type_lower + ' hardware_efficiency'] = str("{:.3f}".format(op_hw_efficiency)) + "%"

    # step 5：将 timeline_data 导出为 .json
    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f)


