import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import  torch
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)


def polynomial_fun(w, x):
    x = x.view(-1, 1)
    powers_of_x = x**torch.arange(len(w)).view(1, -1)
    y = torch.mm(powers_of_x, w)

    return y

def fit_polynomial_ls(x, t, M = 0):
    x = x.view(-1, 1)
    t = t.view(-1, 1)

    powers_of_x = x**torch.arange(M+1, dtype=torch.float32)

    w = torch.linalg.lstsq(powers_of_x, t).solution

    return w.view(-1)

def fit_polynomial_sgd(x, t, M = 0, lr=1e-2, miniBatchSize=5, print_freq=100, N_epochs=500):
    x = x.view(-1, 1)
    t = t.view(-1, 1)
    powers_of_x = x**torch.arange(M + 1).view(1, -1)
    
    model = torch.nn.Linear(M + 1, 1, bias=False)
    criterion = torch.nn.MSELoss()
    #adagrad
    opt = torch.optim.Adagrad(model.parameters(), lr=0.1)
    dataset = TensorDataset(powers_of_x, t)
    loader = DataLoader(dataset, batch_size=miniBatchSize, shuffle=True)
    
    for epoch in range(N_epochs):
        for batch_x, batch_t in loader:
            opt.zero_grad()
            batch_y = model(batch_x)
            #mean square loss
            loss = torch.nn.MSELoss(batch_y, batch_t)
            loss.backward()
            opt.step()
            if (epoch + 1) % print_freq == 0:
                print('epoch {} loss {}'.format(epoch+1, loss.item()))
                
    return model.weight.data.view(-1)

#implement task script 1

#1) Use polynomial_fun (𝑀 = 2, 𝐰 = [1,2,3]T) to generate a training set and a test set
w = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
x_train = torch.linspace(-20, 20, 20)
x_test = torch.linspace(-20, 20, 10)
y_train = polynomial_fun(w, x_train)
y_test = polynomial_fun(w, x_test)
#observed t values with gauss noise 
t_train = y_train + 0.5 * torch.randn_like(y_train)
t_test = y_test + 0.5 * torch.randn_like(y_test)

#2) Use fit_polynomial_ls (𝑀𝜖{2,3,4}) to compute the optimum weight 

M = torch.tensor([2,3,4])

#For each 𝑀, compute the predicted target values 𝑦̂ for all 𝑥 in both the training and test sets.
pred_train = torch.zeros(len(x_train), len(M))
pred_test = torch.zeros(len(x_test), len(M))
for i, m in enumerate(M):
    w_hat = fit_polynomial_ls(x_train,t_train,m.item()).unsqueeze(1)
    pred_train[:, i] = polynomial_fun(w_hat, x_train).squeeze()
    pred_test[:, i] = polynomial_fun(w_hat, x_test).squeeze()
    
# Report, using printed messages, the mean (and standard deviation) in difference a) between
# the observed training data and the underlying “true” polynomial curve; and b) between the
# “LS-predicted” values and the underlying “true” polynomial curve.
#both test and train

diff_obs_train = t_train - y_train
diff_pred_train = pred_train - y_train
diff_obs_test = t_test - y_test
diff_pred_test = pred_test - y_test

sd_obs_train, mean_obs_train = torch.std_mean(diff_obs_train)
sd_pred_train, mean_pred_train = torch.std_mean(diff_pred_train)
sd_obs_test, mean_obs_test = torch.std_mean(diff_obs_test)
sd_pred_test, mean_pred_test = torch.std_mean(diff_pred_test)


print('Training Mean difference between the observed training data and the underlying “true” polynomial curve is ', mean_obs_train.item(), 'and standard diviation is', sd_obs_train.item() )
print('Training Mean difference between “LS-predicted” values and the underlying “true” polynomial curve is {:.3f}, and standard diviation: {:.3f}' .format(mean_pred_train.item(), sd_pred_train.item()) )
print('Test Mean difference between the observed training data and the underlying “true” polynomial curve in training data is ', mean_obs_test.item(), 'and standard diviation is', sd_obs_test.item())
print('Test  Mean difference between “LS-predicted” values and the underlying “true” polynomial curve is {:.3f}, and standard diviation: {:.3f}' .format(mean_pred_test.item(), sd_pred_test.item()) )

    
#Use fit_polynomial_sgd (𝑀𝜖{2,3,4}) to optimise the weight vector 𝐰̂ using the training set.
# For each 𝑀, compute the predicted target values 𝑦̂ for all 𝑥 in both the training and test sets.
#  Report, using printed messages, the mean (and standard deviation) in difference between the
# “SGD-predicted” values and the underlying “true” polynomial curve
pred_train_opt = torch.zeros(len(x_train), len(M))
pred_test_opt = torch.zeros(len(x_test), len(M))
for i, m in enumerate(M):
    w_hat_opimized = fit_polynomial_sgd(x_train,t_train,m.item()).unsqueeze(1)
    pred_train_opt[:, i] = polynomial_fun(w_hat, x_train).squeeze()
    pred_test_opt[:, i] = polynomial_fun(w_hat, x_test).squeeze()

diff_pred_train_opt = pred_train_opt - y_train
diff_pred_test_opt = pred_test_opt - y_test

sd_pred_train_opt, mean_pred_train_opt = torch.std_mean(diff_pred_train_opt)
sd_pred_test_opt, mean_pred_test_opt = torch.std_mean(diff_pred_test_opt)

print('Training Mean difference between “SGD-predicted” values and the underlying “true” polynomial curve is {:.3f}, and standard diviation: {:.3f}' .format(mean_pred_train_opt.item(), sd_pred_train_opt.item()) )
print('Test  Mean difference between “SGD-predicted” values and the underlying “true” polynomial curve is {:.3f}, and standard diviation: {:.3f}' .format(mean_pred_test_opt.item(), sd_pred_test_opt.item()) )

#using sgd, approximate your M using continous number of variables, to optimize M
#consider how you are updating your optimized, if it's not optimum what can u do about it 