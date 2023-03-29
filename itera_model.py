# -*- coding: utf-8 -*-
# 这一段是实体概念扩展

# 将test文件换为dataset_testall文件, fewshot.py第404行换为dataset_prelabel_0.csv, 第539行换为去掉注释


import logging
import sys
import subprocess
import time
import os

if __name__ == '__main__':
    file1 = open("news6_testall.csv", "r")
    file2 = open("./datasets/TextClassification/newsgroups6/test.csv", "w")

    m = file1.read()
    n = file2.write(m)

    file1.close()
    file2.close()

    for i in range(1, 11):
        file3 = open("news6_train" + str(i) + ".csv", "r")
        file4 = open("./datasets/TextClassification/newsgroups6/train.csv", "w")

        s = file3.read()
        w = file4.write(s)

        file3.close()
        file4.close()

        cmd = "python fewshot.py --result_file ./output_fewshot.txt --dataset newsgroups6  --template_id 3  --seed 130 --shot 50 --verbalizer manual"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()

        file5 = open("news6_prelabel_0.csv", "r")
        file6 = open("news6_prelabel" + str(i) + ".csv", "w")

        a = file5.read()
        b = file6.write(a)

        file5.close()
        file6.close()
        print("train {}:preprocess down".format(i))

        time.sleep(3)




