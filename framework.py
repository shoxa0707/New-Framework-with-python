import random
import math
import numpy as np
import time
import json

class Layer:
    def __init__(self, inp=1, out=1):
        self.in_neuron = inp
        self.out_neuron = out
        self.create_parameters()
    
    def create_parameters(self):
        self.weights = np.random.uniform(-1, 1, (self.in_neuron, self.out_neuron))
        self.bias = np.random.uniform(-1, 1, (1, self.out_neuron))
        
    def set_parameters(self, x):
        self.weights = np.array(x['weights'])
        self.bias = np.array(x['bias'])
        
    def get_parameters(self):
        return [self.weights, self.bias]
    
    def get_data(self):
        return [self.in_neuron, self.out_neuron, self.name]
        
    def __call__(self,x):
        if isinstance(x, list):
            self.input = x[0]
        else:
            self.input = x
        out = [np.dot(self.input, self.weights) + self.bias, self.name]
        self.output = out.copy()
        return out
        
class Activations:
    def __init__(self,act_name='relu'):
        if act_name == 'relu':
            self.active_func = self.relu
            self.derivate = self.der_relu
        elif act_name == 'sigmoid':
            self.active_func = self.sigmoid
            self.derivate = self.der_sigmoid
        elif act_name == 'softmax':
            self.active_func = self.softmax
            self.derivate = self.der_softmax
        else:
            print('bunaqa activation function bizda yo\'q')
        self.output = []
        self.name = act_name
        
    def get_data(self):
        return self.name
    
    def relu(self, xname):
        x = xname[0]
        out = [np.maximum(0,x), xname[1], self.name]
        self.output.append(out.copy())
        return out
    
    def der_relu(self, a):
        x = a.copy()
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def sigmoid(self,xname):
        x = xname[0]
        out = [1/(1+np.exp(-x)), xname[1], self.name]
        self.output.append(out.copy())
        return out
    
    def der_sigmoid(self, x):
        return x*(1-x)
    
    def softmax(self, xname):
        x = xname[0]
        soft = []
        for i in x:
            soft.append(np.exp(i)/np.sum(np.exp(i)))
        out = [np.array(soft), xname[1], self.name]
        self.output.append(out.copy())
        return out
    
    def der_softmax(self, x):
        return np.diag(x) - np.outer(x, x)
    
    def __call__(self,x):
        return self.active_func(x)
    
class Loss:
    def __init__(self, loss_name='MSE'):
        if loss_name == 'MSE':
            self.loss_func = self.MSE
            self.derivate = self.der_MSE
        elif loss_name == 'MAE':
            self.loss_func = self.MAE
            self.derivate = self.der_MAE
        elif loss_name == 'cross_entropy_loss':
            self.loss_func = self.CEL
            self.derivate = self.der_CEL
        else:
            print('bunaqa loss bizda yo\'q')
        self.name = loss_name
    
    def MSE(self,target, predict):
        return (target-predict)**2
    
    def der_MSE(self, target, predict):
        return -2*(target-predict)
    
    def MAE(self, target, predict):
        return abs(target-predict)
    
    def der_MAE(self, target, predict):
        if target > predict:
            return -1
        elif target == predict:
            return 0
        else:
            return 1
        
    def CEL(self, target, predict):
        return -np.sum(np.multiply(target,np.log(predict)))
    
    def der_CEL(self, target, predict):
        return predict - target
    
    def __call__(self, target, output):
        return self.loss_func(target, output)

class PyDahoShoxa(object):
    params = {}
    layers_list = []
    activs_list = []
    def __init__(self, layers=[]):
        self.layers = layers
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.optimizer.loss = Loss(loss)
    
    def forward(self, x):
        self.layers_out = [x]
        for i in range(len(self.layers)):
            x = np.dot(x, self.layers[i].weights) + self.layers[i].bias
            self.layers_out.append(x)
        return x
    
    def predict(self, x):
        return self.forward(x)[0][0]
    
    def parameters(self):
        parameters = []
        for i in self.layers_list:
            parameters.append(self.params[i].get_parameters())
        return parameters
    
    def set_parameters(self, updated_param):
        for ind, i in enumerate(self.layers_list):
            self.params[i].set_parameters(updated_param[ind])
            
    def get_outputs(self):
        outs = []
        for i in self.layers_list:
            outs.append(self.params[i].output)
        names = [i[1] for i in outs]
        for i in self.activs_list:
            for j in self.params[i].output:
                if j[1] in names:
                    ind = names.index(j[1])
                    outs.pop(ind)
                    outs.insert(ind, j)
        out = outs.copy()
        if len(self.sequence) == 0 and len(out) > 0:
            for i in out:
                for j in i[1:]:
                    self.sequence.append(self.params[j].get_data())
        return out
    
    def fit(self, x, y, batch_size=4, epochs=10):
        self.optimizer.batch_size = batch_size
        count_batch = math.ceil(len(x)/batch_size)
        for epoch in range(epochs):
            jami = time.time()
            np.random.shuffle(x)
            acc = 0
            print(f"Epoch {epoch+1}/{epochs}")
            batches = self.split_datas(x=x, y=y, batch_size=batch_size, count_batch=count_batch)
            for ind, batch in enumerate(batches):
                start = time.time()
                self.forward(batch[0])
                outputs = self.get_outputs()
                outputs.insert(0, [batch[0]])
                self.optimizer.outputs = outputs.copy()
                self.optimizer.backward(batch[1])
                loss = self.optimizer.loss(batch[1], outputs[-1][0][0])
                if self.sequence[-1] == 'sigmoid' or self.optimizer.loss.name == 'cross_entropy_loss':
                    acc += self.accuracy(batch[1], outputs[-1][0])
                    acc /= len(x)
                else:
                    acc = self.accuracy(batch[1], outputs[-1][0])
                if ind != count_batch - 1:
                    eta = self.sec2user(math.floor((count_batch-ind-1)*(time.time()-start)))
                    about = f"""{ind+1}/{count_batch} [{int((ind+1)/count_batch*30)*"="}>{(29-int((ind+1)/count_batch*30))*"."}] - ETA: {eta} - loss: {round(loss, 4)} - accuracy: {round(acc, 4)}"""
                    print(about,end='\r')
                else:
                    jami = time.time()-jami
                    about = f"""{(ind+1)}/{count_batch} [{30*"="}] - {self.sec2user(math.floor(jami))} {math.floor(jami/count_batch)}ms/step - loss {round(loss, 4)} - accuracy: {round(acc, 4)}"""
                    print(about)
    
    def sec2user(self, sec):
        secs = str(sec % 60)
        if sec < 60:
            return secs + 's'
        if len(secs) == 1:
            secs = '0'+str(sec)
        while True:
            sec = sec // 60
            if sec >= 60:
                qism = sec% 60
                if qism < 10:
                    secs = '0'+str(qism)+':'+secs
                else:
                    secs = str(qism)+':'+secs
            else:
                if sec < 10:
                    secs = '0'+str(sec)+':'+secs
                else:
                    secs = str(sec)+':'+secs
                break
        return secs
    
    def split_datas(self, x, y, batch_size=4, count_batch=0):
        self.optimizer.batch_size = batch_size
        datas = [(x[i],y[i]) for i in range(len(x))]
        random.shuffle(datas)
        for i in range(count_batch):
            a = datas[i*batch_size:(i+1)*batch_size]
            yield np.array([k[0] for k in a]), np.array([k[1] for k in a])
                    
    def evaluate(self, x, y):
        start = time.time()
        predict = self.forward(x)
        return self.r2_score(y, predict[0])
    
    def accuracy(self, target, predict):
        if self.optimizer.loss.name == 'cross_entropy_loss':
            predict = np.argmax(predict, axis=1)
            return np.sum(target==predict)
        elif self.sequence[-1] == 'sigmoid':
            predict[predict>=0.5] = 1
            predict[predict<0.5] = 0
            return np.sum(target==predict)
        else:
            self.r2_score(target, predict)
    
    def r2_score(self, target, predict):
        tar = target[:len(predict)]
        residuals = np.sum(Loss('MSE')(tar, predict))
        mean = np.mean(target)
        total = np.sum(np.power(target - mean, 2))
        return 1 - residuals/total
            
    def save(self, path):
        model = {}
        pivot = 1
        for i in self.sequence:
            if isinstance(i, list):
                parameters = self.params[i[2]].get_parameters()
                model[f"Layer{pivot}"] = {}
                model[f"Layer{pivot}"]['weights'], model[f"Layer{pivot}"]['bias'] = parameters[0].tolist(), parameters[1].tolist()
                pivot += 1
                
        model['Architecture'] = self.sequence
        model['Compile'] = {}
        model['Compile']['optimizer_learning_rate'] = self.optimizer.learning_rate
        model['Compile']['loss'] = self.optimizer.loss.name
        with open(path, 'w') as f:
            json.dump(model, f, indent=4)
    
    def __setattr__(self, name, value):
        self.params[name]=value
        if isinstance(value, Layer):
            if name not in self.layers_list:
                self.params[name].name = name
                self.layers_list.append(name)
        elif isinstance(value, Activations):
            if name not in self.activs_list:
                self.params[name].name = name
                self.activs_list.append(name)
        super().__setattr__(name, value)
        
    def __new__(cls):
        obj = super().__new__(cls)
        obj.params = {}
        obj.layers_list = []
        obj.activs_list = []
        obj.sequence = []
        return obj
    
    def __call__(self,x):
        return self.forward(x)
    
class Optimizer:
    def __init__(self, params=[], outputs=[], actives=['relu','sigmoid','softmax'], learning_rate=0.001):
        self.params = params
        self.outputs = outputs.copy()
        self.learning_rate = learning_rate
        self.activations = {}
        for i in actives:
            self.activations[i] = Activations(i)
    
    def qism_gradient(self, order, satr, ustun, batch):
        if len(self.outputs[-1]) > 2:
            if self.outputs[-1][2] == 'softmax':
                if order == len(self.params) - 1:
                    last_act = np.zeros((self.outputs[-1][0][batch].shape))
                    last_act[ustun] = 1.0
                    return last_act
                else:
                    last_act = 1
            else:
                last_act = self.activations[self.outputs[-1][-1]].derivate(self.outputs[-1][0][batch])
        else:
            last_act = 1
        if order == len(self.params) - 1:
            return last_act
        else:
            grad = self.params[order+1][0][[ustun]].copy()
            for i in range(order+2, len(self.params)):
                if len(self.outputs[i]) > 2:
                    acts_matrix = np.tile(self.activations[self.outputs[i][-1]].derivate(self.outputs[i][0][batch])[0], (self.params[i][0].shape[1], 1)).T
                else:
                    acts_matrix = np.ones(self.params[i][0].shape)
                layer_next = np.multiply(self.params[i][0], acts_matrix)
                del acts_matrix
                grad = np.dot(grad, layer_next)
            result = np.multiply(grad, last_act)
            del grad, last_act
            return result
    
    def backward(self, target):
        common_der_loss = self.loss.derivate(target, self.outputs[-1][0])
        # layerlar uchun sikl
        for i in range(len(self.params)-1, -1, -1):
            ###################################
            #### weightlarni update qilish ####
            ###################################
            qism_grad = []
            if len(self.outputs[i]) == 3 and i != len(self.params)-1 and self.outputs[i][2] != 'softmax':
                act_common = self.activations[self.outputs[i][-1]].derivate(self.outputs[i][0])
                common = []
                for son in range(len(act_common)):
                    common.append(np.tile(act_common[son], (self.params[i][0].shape[1], 1)).T)
                act_common = np.array(common)
                del common
            else:
                act_common = np.ones((self.batch_size, self.params[i][0].shape[0], self.params[i][0].shape[1]))
            for satr in range(len(self.params[i][0])):
                for ustun in range(len(self.params[i][0][satr])):
                    try:
                        umumiy_grad = 0
                        for batch in range(self.batch_size):
                            umumiy_grad += np.dot(qism_grad[ustun][batch], common_der_loss[batch])*act_common[batch][satr, ustun]*self.outputs[i][0][batch][satr]
                        self.params[i][0][satr,ustun] -= self.learning_rate*(umumiy_grad/self.batch_size)
                        del umumiy_grad
                    except:
                        qism_grad.append([])
                        umumiy_grad = 0
                        for batch in range(self.batch_size):
                            qism_grad[-1].append(self.qism_gradient(i, satr, ustun, batch))
                            umumiy_grad += np.dot(qism_grad[ustun][batch], common_der_loss[batch])*act_common[batch][satr, ustun]*self.outputs[i][0][batch][satr]
                        self.params[i][0][satr,ustun] -= self.learning_rate*(umumiy_grad/self.batch_size)
                        del umumiy_grad
            #################################
            #### biaslarni update qilish ####
            #################################
            for bias in range(len(qism_grad)):
                umumiy_grad = 0
                for batch in range(self.batch_size):
                    umumiy_grad += np.dot(qism_grad[ustun][batch], common_der_loss[batch])*act_common[batch][satr, ustun]
                self.params[i][1][0,bias] -= self.learning_rate*(umumiy_grad/self.batch_size)
                del umumiy_grad
            del qism_grad
        del common_der_loss

    
    
    
def load_model(path):
    model = PyDahoShoxa()
    # read model parameters and architecture
    with open(path) as f:
        model_data = json.load(f)
    model.sequence = model_data['Architecture']
    # set layers and their parameters
    lay = 1
    for i in model.sequence:
        if isinstance(i, list):
            model.layers_list.append(i[2])
            model.params[i[2]] = Layer(i[0], i[1])
            model.params[i[2]].name = i[2]
            model.params[i[2]].set_parameters(model_data[f'Layer{lay}'])
            setattr(model, i[2], model.params[i[2]])
            lay += 1
        else:
            model.activs_list.append(i)
            model.params[i] = Activations(i)
    model.compile(Optimizer(params=model.parameters(), learning_rate=model_data['Compile']['optimizer_learning_rate']), loss=model_data['Compile']['loss'])

    def forward(x):
        for i in model.sequence:
            if isinstance(i, list):
                x = model.params[i[2]](x)
            else:
                x = model.params[i](x)
        return x
    
    model.forward = forward

    return model