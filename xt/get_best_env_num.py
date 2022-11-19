import datetime
import os
import subprocess
import numpy as np
import yaml

def line_fitting(X, Y, y):
    # X=np.array([1, 2, 4, 8, 16]) 
    # Y=np.array([45, 62, 116, 267, 573])

    #定义直线拟合函数
    def linear_regression(x, y): 
        N = len(x)
        sumx = sum(x)
        sumy = sum(y)
        sumx2 = sum(x**2)
        sumxy = sum(x*y)
    
        A = np.mat([[N, sumx], [sumx, sumx2]])
        b = np.array([sumy, sumxy])
    
        return np.linalg.solve(A, b)
    
    a10, a11 = linear_regression(X, Y)
    
    y = 653.9
    x = (y - a10) / a11
    return x

def create_yaml(file_name, update_dic, abs_path, str_time):
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    
    doc["env_para"]["env_info"]["size"] = update_dic["size"]
    doc["env_para"]["env_info"]["wait_nums"] = update_dic["wait_nums"]

    doc["env_num"] = update_dic["env_num"]

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    
    doc["benchmark"].update({"id": str_ + "_" + str_time})
    doc["benchmark"].update({"archive_root": doc["benchmark"]["archive_root"]})

    with open(abs_path + doc["benchmark"]["id"] + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

def create_yaml_LR(file_name, update_dic, abs_path):
    with open(file_name) as f:
        doc = yaml.safe_load(f)

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    doc["benchmark"].update({"id": str_})

    doc["model_para"]["actor"]["model_config"].update(update_dic)
    with open(abs_path + doc["benchmark"]["id"] + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

def get_env_num(file_name):
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    return doc["env_num"]

def create_time_record_yaml(file_name, str_time):
    file_dir = file_name.rsplit("/", 1)[0]
    file_save_path = file_dir + "/test_" + str_time
    if not os.path.exists(file_save_path):
        os.mkdir(file_dir + "/test_" + str_time)
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    
    test_complete_step = 100000
    archive_root = doc["benchmark"]["archive_root"] + "_record_time_" + str_time
    doc["benchmark"].update({"archive_root": archive_root})

    for i in range(4):
        env_num = pow(2, i + 2)
        doc["env_num"] = env_num
        doc["agent_para"]["agent_config"].update({"complete_step": test_complete_step})
        with open(file_save_path + "/env_" + str(env_num) + "__" + str_time + ".yaml", 'w') as f:
            yaml.safe_dump(doc, f, default_flow_style=False)

    record_file_path = doc["benchmark"]["archive_root"]

    return file_save_path, record_file_path


def get_time_from_record_txt():
    with open("/home/xys/xt_logs/test_record_time/good_boy_record_time_2022-11-11_22:41:14", "r") as f:
        f.readlines()

def get_record_files_by_path(path):
    dirs = []
    files = []
    for item in os.scandir(path):
        if item.is_dir():
            dirs.append(item.path)
        elif item.is_file():
            files.append(item.path)
    return files

def get_env_num_and_explorer_and_train_time(files):
    prepare_times_per_time = []
    mean_explorer_times = []
    mean_train_times = []

    for file in files:
        filename = file.rsplit("/", 1)[-1]
        env_num = filename.split(".")[0]
        prepare_times_per_time.append(int(env_num))
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []
        total_skip = 2
        skip_exp = 0
        skip_train = 0
        with open(file, "r") as f:
            lines=f.readlines()
            for line in lines:  
                # print(line)
                if line == "\n":
                    continue 
                key = line.split("\t")[0]
                val = line.split("\t")[1].strip()
                if key.startswith("mean_explore_ms") and val != "nan":
                    if skip_exp < total_skip:
                        skip_exp += 1
                        continue
                    mean_explorer_times_this_env_num.append(float(val))
                if key.startswith("mean_train_time_ms") and val != "nan":
                    if skip_train < total_skip:
                        skip_train += 1
                        continue
                    # if float(val) < 100:
                    mean_train_times_this_env_num.append(float(val))

        mean_explorer_times.append(np.mean(mean_explorer_times_this_env_num))
        mean_train_times.append(np.mean(mean_train_times_this_env_num))

        print(mean_explorer_times)
        print(mean_train_times)
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []

    return np.array(prepare_times_per_time), mean_explorer_times, np.array(mean_train_times)


def run_multi_job_with_dif_env_sync(file_name):
    str_time = str(datetime.datetime.now().strftime('%F_%T'))
    # 创建env_num=1,2,4,8,16并且complete_step=50000的yaml文件 
    # file_save_path为生成的yaml文件夹 其路径和输入的yaml文件的路径在同一级目录
    # record_file_path为生成的目录 其值为输入yaml文件的achieve_root+str_time \ 
    # 其中包含${env_num}.txt以及对应env_num数量生成的benchmark目录
    file_save_path, record_file_path = create_time_record_yaml(file_name, str_time) #  yaml_created_path, achieve root
    os.system("/bin/bash /home/xys/xingtian-ppo-v1/train_yaml/common_train.sh " + file_save_path)
    files = get_record_files_by_path(record_file_path)
    prepare_times_per_time, mean_explorer_times, mean_train_times = get_env_num_and_explorer_and_train_time(files)
    print("prepare_times_per_time ========= {}".format(prepare_times_per_time))
    balance_env_num = line_fitting(prepare_times_per_time, mean_train_times, mean_explorer_times)
    print(balance_env_num)
    # os.system("rm -rf " + file_save_path)


def run_multi_job_with_dif_env_async(file_name):
    str_time = str(datetime.datetime.now().strftime('%F_%T'))
    # 创建env_num=1,2,4,8,16并且complete_step=50000的yaml文件 
    # file_save_path为生成的yaml文件夹 其路径和输入的yaml文件的路径在同一级目录
    # record_file_path为生成的目录 其值为输入yaml文件的achieve_root+str_time \ 
    # 其中包含${env_num}.txt以及对应env_num数量生成的benchmark目录
    file_save_path, record_file_path = create_time_record_yaml(file_name, str_time) #  yaml_created_path, achieve root
    print("file_save_path ============= {}".format(file_save_path))
    print("record_file_path ============= {}".format(record_file_path))

    os.system("/bin/bash /home/xys/xingtian-ppo-v1/train_yaml/common_train.sh " + file_save_path)
    files = get_record_files_by_path(record_file_path)
    prepare_times_per_train, mean_explorer_times, mean_train_times = get_env_num_and_explorer_and_train_time(files)
    print("prepare_times_per_time ========= {}".format(prepare_times_per_train))
    print("mean_explorer_times ============== {}".format(mean_explorer_times))
    print("mean_train_times ====================={}".format(mean_train_times))
    if len(np.unique(prepare_times_per_train)) == 1:
        balance_env_num = mean_explorer_times[0] / mean_train_times[0] * prepare_times_per_train[0] 
    else:
        balance_env_num = line_fitting(prepare_times_per_train, mean_train_times, mean_explorer_times)
    print(balance_env_num)
    # os.system("rm -rf " + file_save_path)

    
def create_yaml_dif_env(file_name, update_dic, abs_path, str_time):
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    

    doc["env_num"] = update_dic["env_num"]

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    
    doc["benchmark"].update({"id": str_ + "_" + str_time})
    doc["benchmark"].update({"archive_root": doc["benchmark"]["archive_root"]})

    with open(abs_path + doc["benchmark"]["id"] + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

def create_yaml():
    abs_path = "/home/xys/xingtian-test/xingtian-master3/xingtian-master/train_yaml/train_dif_env/"
    filename = "/home/xys/xingtian-test/xingtian-master3/xingtian-master/train_yaml/breakout_impala.yaml"
    str_time = str(datetime.datetime.now().strftime('%F_%T'))
    update_arrs = []
    start_env = 26
    skip_env = 2
    end_env = 40
    for i in range(start_env, end_env, skip_env):
        update_dic = {"env_num": i}
        create_yaml_dif_env(filename, update_dic, abs_path, str_time)


if __name__ == "__main__":
    
    # create_yaml()
    run_multi_job_with_dif_env_async("/home/xys/xingtian-test/xingtian-master3/xingtian-master/record_time/breakout_impala.yaml")
    # get_time_from_record_txt()
    # record_file_path = "/home/xys/xt_logs/test_record_time/good_boy_record_time_2022-11-11_23:28:36"
    # files = get_record_files_by_path(record_file_path)
    # print(files)
    # run_multi_job_with_dif_env()
    # X = np.array([16, 4, 8, 32])
    # # X = [466.0774716000001, 430.73788415, 452.45377369999994, 493.229166125]
    # Y = np.array([522.0380787368422, 116.14055942105264, 233.80913784210532, 961.9633787999998])
    # y = 452
    # x = line_fitting(X, Y, y)
    # print(x)
    # prepare_times_per_train, mean_explorer_times, mean_train_times = get_env_num_and_explorer_and_train_time(["/home/xys/xt_logs/test_record_time/good_boy_record_time_2022-11-17_21:35:28/1.txt"])
    # print(prepare_times_per_train)
    # print(mean_explorer_times)
    # print(mean_train_times)

    


    

