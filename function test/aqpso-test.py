import copy

import numpy as np
from cec2017.functions import f5
# x = np.random.uniform(-100, 100, size=50)
# val = f5(x)
# print('f5(x) = %.6f' %val)

import re
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import random
import math
import matplotlib.pyplot as plt
from data import load_data
import time

start_time = time.time()
sum = 0

## 1.加载数据


## 2. QPSO算法
class QPSO(object):
    def __init__(self, particle_num, particle_dim, alpha_max, alpha_min, iter_num, max_value, min_value):
        '''定义类参数
        particle_num(int):粒子群大小
        particle_dim(int):粒子维度，对应待寻优参数的个数
        alpha_max(float):最大控制系数
        alpha_min(float):最小控制系数
        alpha(float):控制系数
        iter_num(int):最大迭代次数
        max_value(float):参数的最大值
        min_value(float):参数的最小值
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_value = max_value
        self.min_value = min_value

    ### 2.1 粒子群初始化
    def swarm_origin(self):
        '''初始化粒子群中的粒子位置
        input:self(object):QPSO类
        output:particle_loc(list):粒子群位置列表
        '''
        particle_loc = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                a = random.random()
                tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
            particle_loc.append(tmp1)

        return particle_loc

    ### 2.2 计算适应度函数数值列表
    def fitness(self, particle_loc):
        '''计算适应度函数值
        input:self(object):PSO类
              particle_loc(list):粒子群位置列表
        output:fitness_value(list):适应度函数值列表
        '''
        fitness_value = []
        ### 1.适应度函数为RBF_SVM的3_fold交叉校验平均值
        for i in range(self.particle_num):
            # rbf_svm = svm.SVC(kernel='rbf', C=particle_loc[i][0], gamma=particle_loc[i][1])
            # cv_scores = model_selection.cross_val_score(rbf_svm, trainX, trainY, cv=7, scoring='accuracy')
            val = f5(particle_loc[i])
            fitness_value.append(val)
        ### 2. 当前粒子群最优适应度函数值和对应的参数
        # current_fitness = float("inf")
        current_fitness = 99999
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness > fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value, current_fitness, current_parameter

    ### 2.3 粒子位置更新
    def updata(self, particle_loc, gbest_parameter, pbest_parameters, best_fitness, current_iter_num):
        '''粒子位置更新
        input:self(object):QPSO类
              particle_loc(list):粒子群位置列表
              gbest_parameter(list):全局最优参数
              pbest_parameters(list):每个粒子的历史最优值
              current_iter_num(int):当前迭代数
        output:particle_loc(list):新的粒子群位置列表
        '''
        Pbest_list = pbest_parameters
        #### 2.3.1 计算mbest
        mbest = []
        total = []
        for l in range(self.particle_dim):
            total.append(0.0)
        total = np.array(total)

        for i in range(self.particle_num):
            total += np.array(Pbest_list[i])
        for j in range(self.particle_dim):
            mbest.append(list(total)[j] / self.particle_num)

        #### 2.3.2 位置更新
        ##### Pbest_list更新
        for i in range(self.particle_num):
            a = random.uniform(0, 1)
            Pbest_list[i] = list(
                np.array([x * a for x in Pbest_list[i]]) + np.array([y * (1 - a) for y in gbest_parameter]))
        ##### particle_loc更新
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (self.iter_num - current_iter_num) / self.iter_num
        for j in range(self.particle_num):
            mbest_x = []  ## 存储mbest与粒子位置差的绝对值
            for m in range(self.particle_dim):
                mbest_x.append(abs(mbest[m] - particle_loc[j][m]))
            u = random.uniform(0, 1)
            if random.random() > 0.5:
                particle_loc[j] = list(
                    np.array(Pbest_list[j]) + np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))
            else:
                particle_loc[j] = list(
                    np.array(Pbest_list[j]) - np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))

        #### 2.3.3 将更新后的量子位置参数固定在[min_value,max_value]内
        ### 每个参数的取值列表
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        ### 每个参数取值的最大值、最小值、平均值
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

        left = []
        right = []
        # 当前所有粒子个体最优位置屮心
        g = []
        # 当前所有粒子位置中心
        n = []
        # 全局最优参数
        p = gbest_parameter
        for i in range(self.particle_dim):
            temp = 0
            for j in range(self.particle_num):
                temp = temp + pbest_parameters[j][i]
            g.append(temp / self.particle_num)

        for i in range(self.particle_dim):
            temp = 0
            for j in range(self.particle_num):
                temp = temp + particle_loc[j][i]
            n.append(temp / self.particle_num)

        for i in range(self.particle_dim):
            p1 = copy.deepcopy(p)
            p2 = copy.deepcopy(p)
            p1[i] = g[i]
            p2[i] = n[i]
            fitness_value = []

            fitness_value.append(best_fitness)
            fitness_value.append(f5(p1))
            fitness_value.append(f5(p2))


            if fitness_value[1] < fitness_value[0]:
                if fitness_value[1] < fitness_value[2]:
                    p = p1
                    best_fitness = fitness_value[1]
                else:
                    p = p2
                    best_fitness = fitness_value[2]
            else:
                if fitness_value[2] < fitness_value[0]:
                    p = p2
                    best_fitness = fitness_value[2]

        current_fitness = best_fitness
        current_parameter = p

        # left.append(g[0])
        # left.append(n[0])
        # left.append(p[0])
        # right.append(g[1])
        # right.append(n[1])
        # right.append(p[1])
        #
        # gbest = []
        # for i in left:
        #     for j in right:
        #         temp = []
        #         temp.append(i)
        #         temp.append(j)
        #         gbest.append(temp)
        #
        #
        # fitness_value = []
        # ### 1.适应度函数为RBF_SVM的3_fold交叉校验平均值
        # for i in range(9):
        #     # rbf_svm = svm.SVC(kernel='rbf', C=gbest[i][0], gamma=gbest[i][1])
        #     # cv_scores = model_selection.cross_val_score(rbf_svm, trainX, trainY, cv=3, scoring='accuracy')
        #     # fitness_value.append(cv_scores.mean())
        #     val = f5(gbest[i])
        #     fitness_value.append(val)
        # ### 2. 当前粒子群最优适应度函数值和对应的参数
        # current_fitness = best_fitness
        # current_parameter = p
        # for i in range(9):
        #     if current_fitness > fitness_value[i]:
        #         current_fitness = fitness_value[i]
        #         current_parameter = gbest[i]

        return particle_loc, current_fitness, current_parameter

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
        # plt.title('AQPSO_RBF_SVM parameter optimization')
        plt.show()

    ## 2.5 主函数
    def main(self):
        # 最佳适应度和平均适应度
        best_results = []
        average_results = []
        best_fitness = 99999
        ## 1、粒子群初始化
        particle_loc = self.swarm_origin()
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
            fitness_value.append(99999)

        ## 3、迭代
        for i in range(self.iter_num):
            ### 3.1 计算当前适应度函数值列表
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            ### 3.2 求当前的gbest_parameter、pbest_parameters和best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] < fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter

            print('iteration is :', i + 1, ';Best parameters:', gbest_parameter, ';Best fitness', best_fitness)
            best_results.append(best_fitness)
            average_results.append(np.mean(current_fitness_value))
            ### 3.3 更新fitness_value
            fitness_value = current_fitness_value
            ### 3.4 更新粒子群
            particle_loc, best_fitness, gbest_parameter = self.updata(particle_loc, gbest_parameter, pbest_parameters, best_fitness, i + 1)
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
    particle_num = 30
    particle_dim = 30
    iter_num = 1000
    alpha_min = 0.1
    alpha_max = 0.6
    max_value = 100
    min_value = -100
    print('----------------3.PSO_RBF_SVM-----------------')
    qpso = QPSO(particle_num, particle_dim, alpha_max,  alpha_min, iter_num, max_value, min_value)
    qpso.main()
