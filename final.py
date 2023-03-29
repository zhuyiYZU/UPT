# -*- coding: utf-8 -*-
# 这一段是实体概念扩展
import logging
import sys
import subprocess
import time
import os

# fewshot.py 注释掉404-408

if __name__ == '__main__':
    # file1 = open("train_all.csv", "r")
    # file2 = open("./datasets/TextClassification/newsgroups6/train.csv", "w")
    #
    # s = file1.read()
    # w = file2.write(s)
    #
    # file1.close()
    # file2.close()

    for i in {110, 120, 130, 140,150}:

        cmd = "python fewshot.py --result_file ./output_fewshot.txt --dataset reuters3  --template_id 0  --seed  " + str(i) + " --shot 20 --verbalizer manual"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()

        print("train seed {}:preprocess down".format(i))

        time.sleep(3)


