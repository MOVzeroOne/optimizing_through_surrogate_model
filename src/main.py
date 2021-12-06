import torch
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
from tqdm import tqdm

"""
when learning a target function and then minimizing the target function with another network. 
You can get stuck in local minima inside of the target function.

"""

class gaussian_act(nn.Module):
    """
    Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs
    """
    def __init__(self,a=0.01):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a)) 
    
    def forward(self,x):
        return torch.exp(-0.5*x**2/self.a**2)



class network(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),gaussian_act(),nn.Linear(hidden_size,hidden_size),gaussian_act(),nn.Linear(hidden_size,output_size))

    def forward(self,x):
        return self.layers(x)

    

def objective_function(x):
    return torch.sin(x*10)*0.5+ x**2 +1

if __name__ == "__main__":
    torch.manual_seed(5)  #0 gets stuck in global optima  , 5 gets stuck in local minima

    """
    objective is to find x such that it minimizes y.
    """
    x = torch.linspace(-2,2,1000) #input
    y = objective_function(x) #output

    approx_net = network(1,1,100)
    optim_net = network(10,1,100)

    optimizer_approx = optim.Adam(approx_net.parameters(),lr=0.001)
    optimizer_optim = optim.Adam(optim_net.parameters(),lr=0.001)
    plt.ion()
    for i in tqdm(range(100000),ascii=True):
        optimizer_approx.zero_grad()
        output = approx_net(x.view(-1,1))
        loss = nn.MSELoss()(output,y.view(-1,1))
        loss.backward()
        optimizer_approx.step()
        if(i >= 100):
            optimizer_optim.zero_grad()

            loss = nn.MSELoss()(approx_net(optim_net(torch.ones(1,10))),torch.zeros(1,1))
            #loss = torch.sum(torch.log(approx_net(optim_net(torch.ones(1,10)))))
            loss.backward()
            optimizer_optim.step()
        with torch.no_grad():
            plt.cla()
            plt.plot(x,y,label="original graph")
            high_res_x = torch.linspace(-2,2,10000).view(-1,1)
            plt.plot(high_res_x,approx_net(high_res_x).view(-1),label="approx graph")
            plt.scatter(x[torch.argmin(y)],y[torch.argmin(y)],color="red",label="optimal")

            chosen_point = optim_net(torch.ones(1,10))
            plt.scatter(chosen_point,objective_function(chosen_point))
            plt.pause(0.1)
    
    plt.show()
