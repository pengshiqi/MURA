from config import opt
from pprint import pprint

root = opt.data_root
train_image_paths = opt.train_image_paths
test_image_paths = opt.test_image_paths

stat = {}
stat2 = {}
im_per_patient = {}

with open(test_image_paths, 'rb') as F:
    d = F.readlines()
    imgs = [root + str(x, encoding='utf-8')[:-1] for x in d]

print(len(imgs))
for img in imgs:
    organ = img.split('/')[6]
    label = img.split('/')[8].split('_')[1]
    if organ in stat.keys():
        if label in stat[organ].keys():
            stat[organ][label] += 1
        else:
            stat[organ][label] = 1
    else:
        stat[organ] = {}
        stat[organ][label] = 1

pprint(stat)

for img in imgs:
    organ = img.split('/')[6]
    patient_id = img.split('/')[7]
    if organ in stat2.keys():
        if patient_id in stat2[organ].keys():
            stat2[organ][patient_id] += 1
        else:
            stat2[organ][patient_id] = 1
    else:
        stat2[organ] = {}
        stat2[organ][patient_id] = 1

for organ in stat2.keys():
    for patient_id in stat2[organ].keys():
        if str(stat2[organ][patient_id]) in im_per_patient.keys():
            im_per_patient[str(stat2[organ][patient_id])] += 1
        else:
            im_per_patient[str(stat2[organ][patient_id])] = 1
        if stat2[organ][patient_id] >= 10:
            print(organ, patient_id, stat2[organ][patient_id])

pprint(im_per_patient)

