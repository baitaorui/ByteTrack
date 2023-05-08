import numpy as np
# def euclidean_distance(list1, list2):
#     result = np.zeros((len(list1), len(list2)))
#     for i, row in enumerate(list1):
#         for j, col in enumerate(list2):
#             result[i, j] = np.linalg.norm(row - col)
#     return result

a = [[255,253],[255,244],[255,220],[256,207],[258,185],[262,180],[267,163],[269,152],[275,151],[277,141],[281,133],[275,129],[285,129],[285,127],[284,123]]
l = []

for ii in range(len(a) - 1) :
    aa = np.asarray(a[ii])
    bb = np.asarray(a[ii + 1])
    l.append(np.linalg.norm(aa - bb))
print(sum(l) /len(l))