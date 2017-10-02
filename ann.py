import numpy as np
import matplotlib.pyplot as plt
import math

class network:
    def __init__(self,n_input=2,n_hidden=5,n_output=2,bias=1,learning_rate=0.00001):
        self.w_input_hidden=np.matrix(np.random.random((n_input+1,n_hidden)))
        self.w_hidden_output=np.matrix(np.random.random((n_hidden+1,n_output)))
        self.d_w_input_hidden=np.matrix(np.random.random((n_input+1,n_hidden)))
        self.d_w_hidden_output=np.matrix(np.random.random((n_hidden+1,n_output)))
        self.hidden=None
        self.bias=bias
        self.learning_rate=learning_rate
        self.error_list_val=[]
        self.error_list_con=[]


def train_gen(n=1000):
    train=[]
    test=[]
    for i in range(n):
            x1=np.random.randint(2,5)
            x2=np.random.randint(6,8)
            b=objective_function(x1,x2)
            c=check_constraints(x1,x2)
            train.append(((x1,x2),b,c))
    for i in range(20):
            x1=np.random.randint(2,5)
            x2=np.random.randint(6,8)
            b=objective_function(x1,x2)
            c=check_constraints(x1,x2)
            test.append(((x1,x2),b,c))
    return train,test


def objective_function(x1,x2):
    return -0.7*math.log(float(x1)/300.0)-0.3*math.log(x2)
def check_constraints(x1,x2):
    if x1+x2<10 and x1>3:
        return 1
    return 0

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))



def feedforward(input,net):
    input = list(input)
    input.append(net.bias)
    input = np.matrix(input)
    net.hidden = map(sigmoid, (input * net.w_input_hidden).tolist()[0])
    net.hidden.append(net.bias)
    net.hidden = np.matrix(net.hidden)
    temp=(input * net.w_input_hidden).tolist()[0]
    output = [temp[0],sigmoid(temp[1])]
    return output

def backpropagation(entry,net):
    input,value_label,constraints_label=entry[0],entry[1],entry[2]
    output=feedforward(input,net)
    err=np.array(output)-np.array([value_label,constraints_label])
    net.error_list_val.append(err[0])
    net.error_list_con.append(err[1])
    delta_con=err[1]*d_sigmoid(output[1])
    delta_val=err[0]*1
    net.d_w_hidden_output=np.matrix([delta_val,delta_con])
    input=list(input)
    input.append(net.bias)
    input=np.array(input)
    t=net.d_w_hidden_output*net.w_hidden_output.T
    err_h=np.array(np.multiply(net.hidden,t))[0][:-1]
    tmp=(np.matrix(map(d_sigmoid,net.hidden.tolist()[0][:-1]))).T
    net.d_w_input_hidden=(np.matrix(err_h))*tmp
    net.w_hidden_output-=(net.learning_rate*net.d_w_hidden_output.T*net.hidden).T
    net.w_input_hidden -= (net.learning_rate * net.d_w_input_hidden * np.matrix(input)).T


train,test=train_gen()
net=network()
avg_val=[]
avg_con=[]
test_val_before=[]
test_con_before=[]
test_val_after=[]
test_con_after=[]


for i in test:
    input,value_label,constraints_label=i[0],i[1],i[2]
    output=feedforward(input,net)
    err=np.array(output)-np.array([value_label,constraints_label])
    test_val_before.append(err[0])
    test_con_before.append(err[1])



for j in range(10): #10 epoches
    net.error_list_con=[]
    net.error_list_val=[]
    for i in train:
        backpropagation(i,net)
    avg_val.append(sum(net.error_list_val)/len(net.error_list_val))#calc average
    avg_con.append(sum(net.error_list_con)/len(net.error_list_con))#calc average

for i in test:
    input,value_label,constraints_label=i[0],i[1],i[2]
    output=feedforward(input,net)
    err=np.array(output)-np.array([value_label,constraints_label])
    test_val_after.append(err[0])
    test_con_after.append(err[1])


plt.title('values error in test before training')
plt.plot(range(len(test_val_before)), test_val_before)
plt.show()

plt.title('Constraints error in test before training')
plt.plot(range(len(test_con_before)), test_con_before)
plt.show()

plt.title('values error during training')
plt.plot(range(len(avg_val)), avg_val)
plt.show()

plt.title('Constraints error during training')
plt.plot(range(len(avg_con)),  avg_con)#, 'bs')
plt.show()

plt.title('values error in test after training')
plt.plot(range(len(test_val_after)), test_val_after)
plt.show()

plt.title('Constraints error in test after training')
plt.plot(range(len(test_con_after)), test_con_after)
plt.show()
