import numpy as np
import matplotlib.pyplot as plt

true_list = np.loadtxt("./true.txt")
false_list = np.loadtxt("./false.txt")
true_list = list(true_list)
false_list = list(false_list)

true_numbers = []
false_numbers = []
true_number = 0
false_number = 0
sum_true=len(true_list)
sum_false=len(false_list)
print(max(true_list))
print(min(false_list))

for i in range(10,50):
    for k in true_list:
        if( k > i):
            true_number = true_number+1
    for k in false_list:
        if( k <= i):
            false_number = false_number +1
    true_numbers.append(true_number/sum_true)  
    false_numbers.append(false_number/sum_false)  
    print(i,true_number,false_number)
    true_number = 0
    false_number = 0

fig = plt.figure()
fig.clf()
ax = plt.subplot(111)
ax.plot([-0.0001,1],[-0.0001,1],'b--',linewidth=2)
ax.plot(false_numbers, true_numbers, 'b*', markersize=10)
ax.plot(false_numbers, true_numbers, 'b-', linewidth=3)
plt.title('DET curve')
plt.xlabel('False Acceptance rate'); plt.ylabel('False Rejection rate')
ax.axis([-0.0001,1,-0.0001,1])
plt.show()


