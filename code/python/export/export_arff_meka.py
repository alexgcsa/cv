import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
import glob, os

path = '/Users/jan/Documents/Git/cv/'


def export_fold(data_set_name):
    # 1) Load arff
    arff = open(path + 'data/arff_meka/' + data_set_name + '.arff', 'r', encoding='ISO-8859-1').read()

    # 2) Split arff into header and body
    head = []
    body = []
    flag = 'head'
    for row, line in enumerate(arff.splitlines()):
        if flag == 'head':
            head.append(line)
        else:
            body.append(line)

        if '@data' in line:
            flag = 'body'

    head = '\n'.join(head) + '\n'

    # 3) Load assignments
    assignment = pd.read_csv(path + 'folds/iterative_order_2/' + data_set_name + '.csv', header=None)[0] - 1

    assert assignment.min() == 0
    assert assignment.max() == 9
    assert len(assignment) == len(body)

    # 4) Generate folds in a loop
    folds = []
    for row in range(10):
        folds.append([])

    for row, value in enumerate(assignment):
        folds[value].append(body[row])

    files = []
    for fold in folds:
        files.append(head + '\n'.join(fold) + '\n')

    # 5) Zip the folds
    filename = path + 'export/iterative_order_2_meka_deflated/' + data_set_name + '-Stratified10FoldsCV-Meka.zip'

    with ZipFile(filename, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zipObj:
        for row, file in enumerate(files):
            zipObj.writestr(zinfo_or_arcname=data_set_name + '-fold' + str(row + 1) + '.arff', data=file)


os.chdir(path + 'folds/iterative_order_2/')
for file in glob.glob('*.csv'):
    data_set_name = file.replace('.csv', '')
    print(data_set_name)
    export_fold(data_set_name)
