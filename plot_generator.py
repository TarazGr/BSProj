import matplotlib.pyplot as plt

def generate_loss_plot(tmp_dir):
    loss = []
    f = open('tmp/log.txt', 'r', encoding='utf8')
    for line in f.readlines():
        line = line.split()[1]
        loss.append(float(line))
    plt.figure(figsize=(13, 13))
    plt.title('Loss plot')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot([(i + 1) * 500 for i in range(len(loss))], loss)
    plt.savefig(tmp_dir + 'loss.png')

def generate_accuracy_plot(tmp_dir):
    accuracy = []
    f = open('tmp/log.txt', 'r', encoding='utf8')
    for line in f.readlines():
        line = line.split()[0]
        accuracy.append(float(line) * 100)
    plt.figure(figsize=(13, 13))
    plt.title('Accuracy plot')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot([(i + 1) * 500 for i in range(len(accuracy))], accuracy)
    plt.savefig(tmp_dir + 'accuracy.png')
