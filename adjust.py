import os

save_path1 = "news6_label0.csv"
read_path1 = 'news6_prelabel0.csv'
f_1 = open(save_path1, 'w', encoding='utf-8')
with open(read_path1, 'r', encoding='utf-8') as f:
    for ii, label in enumerate(f):
        label = int(label)
        label = label + 1
        label = str(label)
        f_1.write('"' + label + '"'+'\n')
    f.close()
f_1.close()

read_path2 = '/home/ubuntu/user_sh/prompt_new/datasets/TextClassification/newsgroups6/test.csv'
f_2 = open("news6_test.csv", 'w', encoding='utf-8')
with open(read_path2, 'r', encoding='utf-8') as t:
    for ii, text in enumerate(t):
        text = text.split(',')
        text = ' '.join(text[1:])
        f_2.write(text)
    t.close()
f_2.close()

with open('news6_label0.csv', 'r', encoding='utf-8') as fa:
    with open('news6_test.csv', 'r', encoding='utf-8') as fb:
        with open('news6_pre0.csv', 'w', encoding='utf-8') as fc:
            for line in fa:
                fc.write(line.strip('\r\n'))
                fc.write(',')
                fc.write(fb.readline())
            fc.close()


save_path2 = "news6_test0.csv"
save_path3 = "news6_testall.csv"
read_path3 = 'news6_pre0.csv'

f_w = open(save_path2, 'w', encoding='utf-8')
t_w = open(save_path3, 'w', encoding='utf-8')
with open(read_path3, 'r', encoding='utf-8') as f:
    for ii, text in enumerate(f):
        text = text.split(',')
        label = text[0]
        text = ' '.join(text[1:]).replace('\n', '')
        if label == '"1"':
            with open(read_path2, 'r', encoding='utf-8') as t:
                for jj, text1 in enumerate(t):
                    if ii == jj:
                        text1 = text1.split(',')
                        label1 = text1[0]
                        text1 = ' '.join(text1[1:]).replace('\n', '')
                        t_w.write(label1 + ',' + text1 + '\n')
            f_w.write(label + ',' + text + '\n')

f_w = open(save_path2, 'a', encoding='utf-8')
t_w = open(save_path3, 'a', encoding='utf-8')
with open(read_path3, 'r', encoding='utf-8') as t:
    for ii, text in enumerate(t):
        # print(text)
        text = text.split(',')
        label = text[0]
        text = ' '.join(text[1:]).replace('\n', '')
        if label == '"2"':
            with open(read_path2, 'r', encoding='utf-8') as t:
                for jj, text1 in enumerate(t):
                    if ii == jj:
                        text1 = text1.split(',')
                        label1 = text1[0]
                        text1 = ' '.join(text1[1:]).replace('\n', '')
                        t_w.write(label1 + ',' + text1 + '\n')
            f_w.write(label + ',' + text + '\n')

os.remove("news6_label0.csv")
os.remove("news6_test.csv")
os.remove("news6_pre0.csv")




