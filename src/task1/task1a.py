import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import  torch
import time
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys

def polynomial_fun(w, x):
    x = x.view(-1, 1)
    powers_of_x = x**torch.arange(len(w)).view(1, -1).to(device)
    y = torch.mm(powers_of_x, w)

    return y.squeeze()


def fit_polynomial_sgd(x, t, M = 0, lr=1e-2, miniBatchSize=5, print_freq=400, N_epochs=3400, reg_param = 0.01):
    #initialise weights to 1
    torch.manual_seed(123)

    x = x.view(-1, 1)
    t = t.view(-1, 1)
    pow_of_x = x**torch.arange(M + 1).view(1, -1).to(device)
    weights = torch.rand(M + 1, 1, requires_grad=True, device=device)

    #print('weights ', model.weight.data)
    #change to sgd
    opt = torch.optim.Adam([weights], lr = lr)
    dataset = TensorDataset(pow_of_x, t)
    loader = DataLoader(dataset, batch_size=miniBatchSize, shuffle=True)
    mse_loss =  torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    min_loss = torch.inf
    loss_diff_threshold = 1e-3
    losses = []
    grad_norms = []
    print('Training with SGD , polynomial degree ', M)
    
    for epoch in range(N_epochs):
        for batch_x, batch_t in loader:
            opt.zero_grad()
            batch_y = batch_x.to(device)@weights.to(device)
            
            loss = mse_loss(batch_y, batch_t.to(device))
            loss += reg_param * torch.norm(weights)
            loss.backward() 
            #TODO: for plotting only
            grad_norms.append(weights.grad.norm().item())

            #torch.nn.utils.clip_grad_value_(weights, 1)
            total_loss += loss.item()
            num_batches += 1
            opt.step()

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        #doesn't d much 
        if abs(min_loss - loss.item()) < loss_diff_threshold:
            print('Early stopping at epoch {} with loss {:.2f}'.format(epoch+1, avg_loss))
            break;
        min_loss = avg_loss

        total_loss = 0.0
        num_batches = 0
        
    #Plotting
    #TODO:remove

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(losses)
    # plt.title('Loss per epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')

    # plt.subplot(1, 2, 2)
    # plt.plot(grad_norms)
    # plt.title('Gradient norm per epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Gradient norm')
    # plt.show()
                
    return weights.detach()

#implement task script 1
if __name__ == '__main__':
    with open('output_task1a.txt', 'w') as f:


        #1) Use polynomial_fun (𝑀 = 2, 𝐰 = [1,2,3]T) to generate a training set and a test set
        w = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1).to(device)

        m_values = [1, 2, 3, 4, 5, 6, 7, 8]
        
        N_runs = 10
        best_ms = torch.zeros(N_runs)

        #grid search
        for j in range(N_runs):
            #generate new data each run
            x_train = torch.linspace(-20, 20, 20).to(device)
            x_test = torch.linspace(-20, 20, 10).to(device)
            y_train = polynomial_fun(w, x_train.to(device)).to(device)
            y_test = polynomial_fun(w, x_test.to(device)).to(device)
            t_train = y_train + 0.5 * torch.randn_like(y_train.to(device)).to(device)
            t_test = y_test + 0.5 * torch.randn_like(y_test.to(device)).to(device)
            
            best_rmse = float('inf')
            best_sd = float('inf')
            best_weights_rmse = float('inf')
            best_m = None
            for i, m in enumerate(m_values):
                w_true = torch.arange(1, m + 2).float().unsqueeze(1)
                #print('For polynomial degree ', m)
                
                #for SGD
                w_hat_opimized = fit_polynomial_sgd(x_train.to(device), t_train.to(device), m)
                #training
                pred_train_opt = polynomial_fun(w_hat_opimized, x_train.to(device)).squeeze()
                
                #rmse train
                diff_pred_train_opt = pred_train_opt - t_train.to(device)
                diff_pred_weights_opt = w_true.view(-1, 1).to(device) - w_hat_opimized
                rmse_train_opt= torch.sqrt(torch.mean(torch.square(diff_pred_train_opt)))
                rmse_weights_opt = torch.sqrt(torch.mean(torch.square(diff_pred_weights_opt)))
                std_dev_train_opt = torch.std(diff_pred_train_opt)
                
                if rmse_train_opt < best_rmse and std_dev_train_opt < best_sd and rmse_weights_opt< best_weights_rmse:
                    best_m = m
                    best_rmse = rmse_train_opt
                    best_sd = std_dev_train_opt
                    best_weights_rmse = rmse_weights_opt
                print('Training RMSE {:.3f}, and standard deviation is {:.3f}'.format( rmse_train_opt.item(), std_dev_train_opt.item()))
                print('Training RMSE for weights {:.3f}'.format( rmse_weights_opt.item()))
            
            best_ms[j] = best_m
            print('For run {} best m is {} ' .format(j, best_m) )
    
        #test
        print(best_ms)
        most_common = torch.mode(best_ms).values.item()
        w_hat_opimized_test = fit_polynomial_sgd(x_test.to(device), t_test.to(device), best_m)
        pred_test_opt = polynomial_fun(w_hat_opimized_test, x_test.to(device)).squeeze()
        diff_pred_weights_opt = w.view(-1, 1).to(device) - w_hat_opimized
        diff_pred_test_opt = pred_test_opt - t_test.to(device)
        rmse_test_opt= torch.sqrt(torch.mean(torch.square(diff_pred_test_opt)))
        rmse_weights_opt = torch.sqrt(torch.mean(torch.square(diff_pred_weights_opt)))
        std_dev_test_opt = torch.std(diff_pred_test_opt)
        print('Best parameter M is {}'.format(best_m))
        print('Test RMSE  {:.3f}, and standard deviation is {:.3f}'.format( rmse_test_opt.item(), std_dev_test_opt.item()))
        print('Test RMSE for weights {:.3f}'.format( rmse_weights_opt.item()))

        fig, ax = plt.subplots()
        ax.plot(x_test.to(device), y_test.to(device), 'o')
        ax.plot(x_test.to(device), pred_test_opt, 'v')
        plt.show()
    sys.stdout = sys.__stdout__