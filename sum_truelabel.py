file1 = open('train_all.csv', 'r', encoding='utf-8')
file2 = open('news6_testall.csv', 'r', encoding='utf-8')

lines1 = file1.readlines()
lines2 = file2.readlines()

s = 0
t = 0
b = 0
for line1 in lines1:
    t += 1
    a = 0
    for line2 in lines2:
        if line1 == line2:
            a += 1
            if a == 1:
                line1 = line1.split(',')
                table = line1[0]
                if table == '"1"':
                    b += 1
                # f_w = open('together.csv', 'a', encoding='utf-8')
                # f_w.write(table + ',' + line1[1])

                s += 1
    else:
        pass

print('Total pre_table:{}'.format(t))
print('Total pre_truetable:{}'.format(s))
print('Total pre_truetable"1":{}'.format(b))
print('Total pre_truetable"2":{}'.format(s-b))
