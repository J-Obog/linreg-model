from simplelearn import LinReg_Model
from random import randint

data_x = []
data_y = []
min_num = 0
max_num  = 200
gen = 400
train_size = round(gen * 0.8)
test_size = gen - train_size

for i in range(gen):
    x = randint(min_num, max_num)
    y = randint(min_num, max_num)
    z = randint(min_num, max_num)
    t = (10*x)+(5*y)+(2*z)
    data_x.append([x,y,z])
    data_y.append(t)

model = LinReg_Model()
model.train(data_x[:train_size], data_y[:train_size])
#model.predict(data_x[-test_size:], data_y[-test_size:])

