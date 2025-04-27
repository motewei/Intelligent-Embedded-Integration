import mnist_loader
import network
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 加载数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    # 参数设置
    net = network.Network([784, 100, 10])  # 增加隐藏层神经元
    epochs = 50
    mini_batch_size = 20
    eta = 2.5

    # 记录损失和准确率
    train_acc_list = []
    test_acc_list = []

    for j in range(epochs):
        net.SGD(training_data, 1, mini_batch_size, eta)  # 每次只跑1个epoch
        # 训练集准确率（将one-hot向量转为数字标签）
        train_acc = net.evaluate([(x, np.argmax(y)) for x, y in training_data]) / len(training_data)
        test_acc = net.evaluate(test_data) / len(test_data)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch {j+1}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

    # 可视化准确率曲线
    plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc')
    plt.plot(range(1, epochs+1), test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

    # 打印部分预测结果
    for i in range(10):
        x, y = test_data[i]
        pred = np.argmax(net.feedforward(x))
        print(f"Sample {i}: Predicted={pred}, True={y}")