import re
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from data import load_data
import time

start_time = time.time()
sum = 0

## 1.加载数据

## 2. PSO优化算法
class PSO(object):
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value):
        '''参数初始化
        particle_num(int):粒子群的粒子数量
        particle_dim(int):粒子维度，对应待寻优参数的个数
        iter_num(int):最大迭代次数
        c1(float):局部学习因子，表示粒子移动到该粒子历史最优位置(pbest)的加速项的权重
        c2(float):全局学习因子，表示粒子移动到所有粒子最优位置(gbest)的加速项的权重
        w(float):惯性因子，表示粒子之前运动方向在本次方向上的惯性
        max_value(float):参数的最大值
        min_value(float):参数的最小值
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1  ##通常设为2.0
        self.c2 = c2  ##通常设为2.0
        self.w = w
        self.max_value = max_value
        self.min_value = min_value

    ### 2.1 粒子群初始化
    def swarm_origin(self):
        '''粒子群初始化
        input:self(object):PSO类
        output:particle_loc(list):粒子群位置列表
               particle_dir(list):粒子群方向列表
        '''
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                b = random.random()
                tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
                tmp2.append(b)
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)

        return particle_loc, particle_dir

    ## 2.2 计算适应度函数数值列表;初始化pbest_parameters和gbest_parameter
    def fitness(self, particle_loc):
        '''计算适应度函数值
        input:self(object):PSO类
              particle_loc(list):粒子群位置列表
        output:fitness_value(list):适应度函数值列表
        '''
        fitness_value = []
        ### 1.适应度函数为RBF_SVM的3_fold交叉校验平均值
        for i in range(self.particle_num):
            rbf_svm = svm.SVC(kernel='rbf', C=particle_loc[i][0], gamma=particle_loc[i][1])
            cv_scores = model_selection.cross_val_score(rbf_svm, trainX, trainY, cv=7, scoring='accuracy')
            fitness_value.append(cv_scores.mean())
        ### 2. 当前粒子群最优适应度函数值和对应的参数
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value, current_fitness, current_parameter

    ## 2.3  粒子位置更新
    def updata(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
        '''粒子群位置更新
        input:self(object):PSO类
              particle_loc(list):粒子群位置列表
              particle_dir(list):粒子群方向列表
              gbest_parameter(list):全局最优参数
              pbest_parameters(list):每个粒子的历史最优值
        output:particle_loc(list):新的粒子群位置列表
               particle_dir(list):新的粒子群方向列表
        '''
        ## 1.计算新的量子群方向和粒子群位置
        for i in range(self.particle_num):
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random() for y in
                  list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            #            particle_dir[i] = self.w * particle_dir[i] + self.c1 * random.random() * (pbest_parameters[i] - particle_loc[i]) + self.c2 * random.random() * (gbest_parameter - particle_dir[i])
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))

        ## 2.将更新后的量子位置参数固定在[min_value,max_value]内
        ### 2.1 每个参数的取值列表
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        ### 2.2 每个参数取值的最大值、最小值、平均值
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)

        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1]) / (value[j][0] - value[j][1]) * (
                            self.max_value - self.min_value) + self.min_value

        return particle_loc, particle_dir

    ## 2.4 画出适应度函数值变化图
    def plot(self, average_results, best_results):
        '''画图
        '''
        X = []
        for i in range(self.iter_num):
            X.append(i + 1)
        # X = np.arange(1, iter_num + 1).astype(dtype=str)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(X, average_results, label='平均适应度')
        plt.plot(X, best_results, label='最佳适应度')
        plt.legend()
        plt.xlabel('迭代次数', size=15)
        plt.ylabel('适应度', size=15)
        # plt.title('PSO_RBF_SVM parameter optimization')
        plt.show()

    ## 2.5 主函数
    def main(self):
        '''主函数
        '''
        # 最佳适应度和平均适应度
        best_results = []
        average_results = []
        best_fitness = 0.0
        ## 1、粒子群初始化
        particle_loc, particle_dir = self.swarm_origin()
        ## 2、初始化gbest_parameter、pbest_parameters、fitness_value列表
        ### 2.1 gbest_parameter
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        ### 2.2 pbest_parameters
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        ### 2.3 fitness_value
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)

        ## 3.迭代
        for i in range(self.iter_num):
            ### 3.1 计算当前适应度函数值列表
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            ### 3.2 求当前的gbest_parameter、pbest_parameters和best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter

            print('iteration is :', i + 1, ';Best parameters:', gbest_parameter, ';Best fitness', best_fitness)
            best_results.append(best_fitness)
            average_results.append(np.mean(current_fitness_value))
            ### 3.3 更新fitness_value
            fitness_value = current_fitness_value
            ### 3.4 更新粒子群
            particle_loc, particle_dir = self.updata(particle_loc, particle_dir, gbest_parameter, pbest_parameters)
        ## 4.结果展示
        # results.sort()
        end_time = time.time()
        print("耗时: {:.2f}秒".format(end_time - start_time))
        self.plot(average_results, best_results)
        print('Final parameters are :', gbest_parameter)



if __name__ == '__main__':
    print('----------------1.Load Data-------------------')
    trainX, trainY = load_data('../data/xss.txt', '../data/normal.txt')
    print('----------------2.Parameter Seting------------')
    particle_num = 20
    particle_dim = 2
    iter_num = 200
    c1 = 2
    c2 = 2
    w = 0.2
    max_value = 100
    min_value = 0.001
    print('----------------3.PSO_RBF_SVM-----------------')
    pso = PSO(particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value)
    pso.main()
