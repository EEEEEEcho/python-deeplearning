from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 网络架构
network = models.Sequential()
# 层一：密集连接的神经层
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 层二：10路的softmax层，返回一个由10个概率值(总和为1)组成的数组,每个概率值表示当前数字图像
# 属于10个数字类别中某一个的概率
network.add(layers.Dense(10, activation='softmax'))

# 编译步骤，填充损失函数、优化器和需要监控的指标
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 数据预处理，将其变换为网络要求的形状，并缩放到所有值在[0,1]区间上
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 训练集上精度为98.83%

# 在测试集上检验性能
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('\ntest_loss:{} , test_acc: {},'.format(test_loss, test_acc))

# 在测试集上的精度为97.92%,比训练集的精度要小很多，这种差距是过拟合造成的。
# 过拟合是指：机器学习模型在新数据上的性能往往比训练数据上要差

