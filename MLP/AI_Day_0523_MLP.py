#第一个MLP实践
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import time
from AI_Day_0523_MLP_Model import MLP_Mnist  as mlpmodel

#1.定义数据预处理变换
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
#2.下载并加载数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           download=True,
                                           transform=mnist_transform)
print('训练集加载完成')

test_dataset = torchvision.datasets.MNIST(root='./data_mnist',
                                          train=False,
                                          download=True,
                                          transform=mnist_transform)
print('测试集加载完成')

#3.创建DataLoader
batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=False)

#4.检查一个批次的数据
data_iter = iter(train_loader)
image_batch, label_batch = next(data_iter)

def imshow_mnist(img_tensor,title=""):
    img_tensor = img_tensor * 0.3081 +0.1307
    npimg = img_tensor.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)).squeeze(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    
# plt.figure(figsize=(10,4))
# for i in range(5):
#     plt.subplot(1,5,i+1)
#     imshow_mnist(image_batch[i], title=f"Label:{label_batch[i].item()}")
# plt.suptitle("Sample MNIST Images from a Batch (After Transforms)", fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

model = mlpmodel()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(device)

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #损失函数
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#设置训练超参数
num_epochs = 5

#训练循环
start_train_time = time.time()

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    correct_predictions_epoch  = 0
    total_samples_epoch = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0) # loss.item()是该批次的平均损失,乘以批大小得到总损失
        
        _, predicted_labels = torch.max(outputs.data, 1) # 获取概率最大类的索引
        total_samples_epoch += labels.size(0)
        correct_predictions_epoch += (predicted_labels == labels).sum().item()
        
        if (batch_idx + 1) % 200 == 0: # 每 200 个批次打印一次训练信息
            print(f'  Batch {batch_idx+1}/{len(train_loader)} | '
                  f'Current Batch Avg Loss: {loss.item():.4f}')
                  
    # 计算并打印当前 epoch 的平均损失和准确率
    epoch_loss = running_loss / total_samples_epoch
    epoch_accuracy = correct_predictions_epoch / total_samples_epoch
    
    print(f"Epoch {epoch+1} 完成: \n"
          f"  平均训练损失 (Avg Training Loss): {epoch_loss:.4f} \n"
          f"  训练准确率 (Training Accuracy): {epoch_accuracy:.4f}")

end_train_time = time.time()
print("\n--- 训练结束 ---")
print(f"总训练时长: {end_train_time - start_train_time:.2f} 秒")