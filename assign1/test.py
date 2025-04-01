# Import from the assignment1.py file
from attack import (
    fgsm_targeted,
    fgsm_untargeted,
    pgd_targeted,
    pgd_untargeted
)
from model import MNIST_Net, CIFAR10_Net
from tqdm.auto import tqdm 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 로드
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=True)
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=False)
    
    return mnist_train_loader, mnist_test_loader, cifar10_train_loader, cifar10_test_loader

def train(model, dataloader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 모델 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def test_adversarial_attacks(model, test_loader, device):
    fgsm_targeted_correct = 0
    fgsm_untargeted_correct = 0
    pgd_targeted_correct = 0
    pgd_untargeted_correct = 0
    total = 0
    
    for images, labels in tqdm(test_loader, desc="Testing adversarial attacks"):
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size
        
        # target label 설정
        # 0-9까지의 label이 존재하므로, 첫 번째 label을 target으로 설정
        target_label = torch.full((batch_size,), labels[0].item(), dtype=torch.long, device=device)
        
        # FGSM Targeted
        perturbed_images = fgsm_targeted(model, images, target_label, eps=0.3)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        fgsm_targeted_correct += (predicted == labels).sum().item()
        
        # FGSM Untargeted
        perturbed_images = fgsm_untargeted(model, images, labels, eps=0.3)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        fgsm_untargeted_correct += (predicted == labels).sum().item()
        
        # PGD Targeted
        perturbed_images = pgd_targeted(model, images, target_label, k=10, eps=0.3, eps_step=0.01)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        pgd_targeted_correct += (predicted == labels).sum().item()
        
        # PGD Untargeted
        perturbed_images = pgd_untargeted(model, images, labels, k=10, eps=0.3, eps_step=0.01)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        pgd_untargeted_correct += (predicted == labels).sum().item()
    
    # 최종 정확도 계산
    print(f"FGSM Targeted Attack Accuracy: {fgsm_targeted_correct/total:.4f} ({fgsm_targeted_correct}/{total})")
    print(f"FGSM Untargeted Attack Accuracy: {fgsm_untargeted_correct/total:.4f} ({fgsm_untargeted_correct}/{total})")
    print(f"PGD Targeted Attack Accuracy: {pgd_targeted_correct/total:.4f} ({pgd_targeted_correct}/{total})")
    print(f"PGD Untargeted Attack Accuracy: {pgd_untargeted_correct/total:.4f} ({pgd_untargeted_correct}/{total})")

# 테스트 실행
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    cifar10_model = CIFAR10_Net().to(device)
    mnist_model = MNIST_Net().to(device)
    
    # dataloader 로드
    mnist_train_loader, mnist_test_loader, cifar10_train_loader, cifar10_test_loader = load_data()
    
    #mnist 모델 train
    epochs = 50
    print("Training MNIST Model...")
    for epoch in range(epochs):
        train_loss = train(mnist_model, mnist_train_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.3f}")
    
    # MNIST 모델 평가
    print("Evaluating MNIST Model...")
    mnist_acc = evaluate(mnist_model, mnist_test_loader, device)
    print(f"MNIST Model Accuracy: {mnist_acc:.3f}")
    
    # adversarial attack 테스트
    print("Testing MNIST Adversarial Attacks...")   
    test_adversarial_attacks(mnist_model, mnist_test_loader, device)
    
    #cifar10 모델 train
    print("Training CIFAR10 Model...")
    for epoch in range(epochs):
        train_loss = train(cifar10_model, cifar10_train_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.3f}")
    
    # CIFAR10 모델 평가
    print("Evaluating CIFAR10 Model...")
    cifar10_acc = evaluate(cifar10_model, cifar10_test_loader, device)
    print(f"CIFAR10 Model Accuracy: {cifar10_acc:.3f}")
    
    # CIFAR10 모델 평가
    print("Testing CIFAR10 Adversarial Attacks...")
    test_adversarial_attacks(cifar10_model, cifar10_test_loader, device)
        
        
    
if __name__ == "__main__":
    main()
