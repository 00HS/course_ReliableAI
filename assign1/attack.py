'''
1 Instruction
• Release date: Monday, March 24, 2025
• Due date: Wednesday, April 2, 2025, by 11:59 PM
• Submission: Upload the project to GitHub and submit the link to Uclass. Late submissions will not be accepted, either through the system or email.
• Important issues
– Reproducibility is crucial for grading. Your GitHub project should include a
requirements.txt that describes all external modules it uses.
– Please include comments in your code to help me understand it.
– Include test.py file to verify the functionality and correctness of your implementation. 
It should train the network sequentially and run an adversarial attack when
run. The specific format for the tests is not required, but they should demonstrate
your code’s validity.

2 Problem Statement
Implement the following tasks using the MNIST and CIFAR-10 datasets. 
For MNIST, build your own model with no specific design requirements, allowing you full flexibility in its
architecture. 
For CIFAR-10, you can either use pre-trained models available online or design your own model. 
If you use pre-trained models or open-source code for CIFAR-10, be sure to properly mention and cite them in your assignment.
1. Implement targeted FGSM with a function signature similar to fgsm targeted(model, x, target, eps). Ensure your implementation clamps the outputs back to the image
domain (i.e., [0, 1]2828 for MNIST). Feel free to modify the signature if necessary.
2. Implement untargeted FGSM. This is similar to the targeted attack, but instead
of passing a target, pass the correct label, i.e fgsm untargeted(model, x, label, eps).
3. Implement iterative, projected FGSM (PGD) for both targeted and untargeted
cases. This algorithm performs k iterations of FGSM with a perturbation magnitude of eps step per iteration. You can build on your previous implementations.
The function signature should resemble pgd targeted(model, x, target, k, eps,
eps step), with the option to extend as needed. Ensure that you clip the output to
the image domain.'''

import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# (1) fsgm_targeted : 공격자가 원하는 target에 맞추어 계산
def fgsm_targeted(model, x, target, eps):
    x.requires_grad = True
    output = model(x)
    #print(output)
    loss = nn.CrossEntropyLoss()(output, target)
    
    model.zero_grad()
    loss.backward()
    
    # 기울기의 부호(grad.sign)를 사용하여 target class에 대한 loss를 증가시키는 방향으로
    perturbed_x = x - eps * x.grad.sign()
    
    # 원본 이미지 도메인 범위로 클램핑 (0-1)
    perturbed_x = torch.clamp(perturbed_x, 0, 1)
    
    return perturbed_x

# (2) fgsm_untargeted
def fgsm_untargeted(model, x, label, eps):
    x.requires_grad = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, label)
    
    model.zero_grad()
    loss.backward()
    
    # 기울기의 부호(grad.sign)를 사용하여 true label를 가장 멀리하는 방향으로
    perturbed_x = x + eps * x.grad.sign()
    
    # 원본 이미지 도메인 범위로 클램핑 (0-1)
    perturbed_x = torch.clamp(perturbed_x, 0, 1)
    
    return perturbed_x

# (3) pgd_targeted
def pgd_targeted(model, x, target, k, eps, eps_step):
    perturbed_x = x.clone().detach()
    
    for i in range(k):
        perturbed_x.requires_grad = True
        
        output = model(perturbed_x)
        loss = nn.CrossEntropyLoss()(output, target)
        
        model.zero_grad()
        loss.backward()
        
        # FGSM 스텝 적용
        # 각 step에서 ε_step만큼 perturbation을 추가
        adv_x = perturbed_x - eps_step * perturbed_x.grad.sign()
        
        # 원본 이미지 근처로 projection
        eta = torch.clamp(adv_x - x, -eps, eps) 
        perturbed_x = torch.clamp(x + eta, 0, 1).detach()
    
    return perturbed_x

# (4) pgd_untargeted
def pgd_untargeted(model, x, label, k, eps, eps_step):
    perturbed_x = x.clone().detach()
    
    for i in range(k):
        perturbed_x.requires_grad = True
        
        output = model(perturbed_x)
        loss = nn.CrossEntropyLoss()(output, label)
        
        model.zero_grad()
        loss.backward()
        
        # FGSM 스텝 적용
        adv_x = perturbed_x + eps_step * perturbed_x.grad.sign()
        
        # 원본 이미지 근처로 projection
        eta = torch.clamp(adv_x - x, -eps, eps)
        perturbed_x = torch.clamp(x + eta, 0, 1).detach()
    
    return perturbed_x

