# input dictionary
d=[
    ('kfold0', 's01_e01'),
    ('kfold0', 's01_e02'),
    ('kfold0', 's02_e01'),
    ('kfold0', 's02_e02'),
    ('kfold2', 's03_e01'),
    ('kfold2', 's03_e02'),
    ('kfold2', 's04_e01'),
    ('kfold2', 's04_e02'),
    ('kfold4', 's05_e01'),
    ('kfold4', 's05_e02'),
    ('kfold4', 's06_e01'),
    ('kfold4', 's06_e02'),
    ('kfold6', 's07_e01'),
    ('kfold6', 's07_e02'),
    ('kfold6', 's08_e01'),
    ('kfold6', 's08_e02'),
    ('kfold8', 's09_e01'),
    ('kfold8', 's09_e02'),
    ('kfold8', 's10_e01'),
    ('kfold8', 's10_e02')
]

# fetch keys
b=[j[0] for i in d for j in i.items()]

# print output
for k in list(set(b)):
    print("{0}: {1}".format(k, b.count(k)))
