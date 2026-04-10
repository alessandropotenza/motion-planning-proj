# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
from mlp import MLPRegression
import os
from cdf import CDF2D
from tqdm import tqdm


class Train_CDF:
    def __init__(self,device) -> None:
        self.device = device  

        self.cdf = CDF2D(device)

        self.q_template = self.cdf.q_grid_template.view(-1,200,2)

        x = torch.linspace(self.cdf.task_space[0][0],self.cdf.task_space[1][0],self.cdf.nbData).to(self.device)
        y = torch.linspace(self.cdf.task_space[0][1],self.cdf.task_space[1][1],self.cdf.nbData).to(self.device)
        xx,yy = torch.meshgrid(x,y)
        xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
        self.p = torch.cat([xx,yy],dim=-1).to(self.device)


    def matching_csdf(self,q):
        # q: [batchsize,2]
        # return d:[len(x),len(q)]
        dist = torch.norm(q.unsqueeze(1).expand(-1,200,-1) - self.q_template.unsqueeze(1),dim=-1)
        d,idx = torch.min(dist,dim=-1)
        q_template = torch.gather(self.q_template,1,idx.unsqueeze(-1).expand(-1,-1,2))
        return d,q_template


    def train(self,input_dim, hidden_dim, output_dim, activate, batch_size, learning_rate, weight_decay, save_path, device,
          epochs):
        # model
        net = MLPRegression(input_dims=input_dim,
                            output_dims=output_dim, 
                            mlp_layers=hidden_dim,
                            skips=[],
                            act_fn=activate, 
                            nerf=True).to(device)
        # net.apply(model.init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        batch_p = self.p.unsqueeze(1).expand(-1,batch_size,-1).reshape(-1,2)
        max_loss = float("inf")
        net.train()
        for i in tqdm(range(epochs)):
            q = torch.rand(batch_size,2,requires_grad=True).to(self.device)*2*torch.pi-torch.pi
            batch_q = q.unsqueeze(0).expand(len(self.p),-1,-1).reshape(-1,2)

            d,q_temp = self.matching_csdf(q)
            q_temp = q_temp.reshape(-1,2)
            mask = d.reshape(-1)<torch.inf
            # mask = d<torch.inf
            # print(d.shape,_p.shape,q.shape)
            inputs = torch.cat([batch_p,batch_q],dim=-1).reshape(-1,4)
            outputs = d.reshape(-1,1)
            inputs,outputs = inputs[mask],outputs[mask]
            q_temp = q_temp[mask]
            weights = torch.ones_like(outputs).to(device)
            # weights = (1/outputs).clamp(0,1)

            d_pred = net.forward(inputs)
            d_grad_pred = torch.autograd.grad(d_pred, batch_q, torch.ones_like(d_pred), retain_graph=True)[0]
            d_grad_pred = d_grad_pred[mask]

            # Compute the Eikonal loss
            eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

            # Compute the MSE loss
            d_loss = ((d_pred-outputs)**2*weights).mean()

            # Compute the projection loss
            proj_q = batch_q[mask] - d_grad_pred*d_pred
            proj_loss = torch.norm(proj_q-q_temp,dim=-1).mean()

            # Combine the two losses with appropriate weights
            w0 = 1.0
            w1 = 1.0
            w2 = 0.1
            loss = w0 * d_loss + w1 * eikonal_loss + w2*proj_loss
            print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}, d_loss: {d_loss.item():.4f}, eikonal_loss: {eikonal_loss.item():.4f}, proj_loss: {proj_loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss.item() < max_loss:
                max_loss = loss.item()
                torch.save(net, os.path.join(save_path, 'model.pth'))

class Train_CDF_EE:
    """Training infrastructure for end-effector-only CDF.

    Same architecture and losses as Train_CDF, but uses data2D_ee.pt
    (configs where the EE tip reaches a workspace point) instead of
    data2D.pt (configs where any link touches a workspace point).
    """

    def __init__(self, device) -> None:
        self.device = device
        self.cdf = CDF2D(device)

        if self.cdf.q_grid_template_ee is None:
            print("End-effector data not found -- generating data2D_ee.pt ...")
            self.cdf.generate_data_ee()
            self.cdf.q_grid_template_ee = torch.load(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data2D_ee.pt')
            )

        self.q_template = self.cdf.q_grid_template_ee.view(-1, 200, 2)

        x = torch.linspace(self.cdf.task_space[0][0], self.cdf.task_space[1][0], self.cdf.nbData).to(self.device)
        y = torch.linspace(self.cdf.task_space[0][1], self.cdf.task_space[1][1], self.cdf.nbData).to(self.device)
        xx, yy = torch.meshgrid(x, y)
        xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
        self.p = torch.cat([xx, yy], dim=-1).to(self.device)

    def matching_csdf(self, q):
        dist = torch.norm(q.unsqueeze(1).expand(-1, 200, -1) - self.q_template.unsqueeze(1), dim=-1)
        d, idx = torch.min(dist, dim=-1)
        q_template = torch.gather(self.q_template, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        return d, q_template

    def train(self, input_dim, hidden_dim, output_dim, activate, batch_size,
              learning_rate, weight_decay, save_path, device, epochs):
        net = MLPRegression(input_dims=input_dim,
                            output_dims=output_dim,
                            mlp_layers=hidden_dim,
                            skips=[],
                            act_fn=activate,
                            nerf=True).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        batch_p = self.p.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, 2)
        max_loss = float("inf")
        net.train()
        for i in tqdm(range(epochs)):
            q = torch.rand(batch_size, 2, requires_grad=True).to(self.device) * 2 * torch.pi - torch.pi
            batch_q = q.unsqueeze(0).expand(len(self.p), -1, -1).reshape(-1, 2)

            d, q_temp = self.matching_csdf(q)
            q_temp = q_temp.reshape(-1, 2)
            mask = d.reshape(-1) < torch.inf
            inputs = torch.cat([batch_p, batch_q], dim=-1).reshape(-1, 4)
            outputs = d.reshape(-1, 1)
            inputs, outputs = inputs[mask], outputs[mask]
            q_temp = q_temp[mask]
            weights = torch.ones_like(outputs).to(device)

            d_pred = net.forward(inputs)
            d_grad_pred = torch.autograd.grad(d_pred, batch_q, torch.ones_like(d_pred), retain_graph=True)[0]
            d_grad_pred = d_grad_pred[mask]

            eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()
            d_loss = ((d_pred - outputs) ** 2 * weights).mean()

            proj_q = batch_q[mask] - d_grad_pred * d_pred
            proj_loss = torch.norm(proj_q - q_temp, dim=-1).mean()

            w0 = 1.0
            w1 = 1.0
            w2 = 0.1
            loss = w0 * d_loss + w1 * eikonal_loss + w2 * proj_loss
            print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}, d_loss: {d_loss.item():.4f}, "
                  f"eikonal_loss: {eikonal_loss.item():.4f}, proj_loss: {proj_loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss.item() < max_loss:
                max_loss = loss.item()
                torch.save(net, os.path.join(save_path, 'model_ee.pth'))


def inference(x,q,net):
    x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,2)
    q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,2)
    inputs = torch.cat([x_cat,q_cat],dim=-1)
    c_dist = net.forward(inputs).squeeze()
    grad = torch.autograd.grad(c_dist, q_cat, torch.ones_like(c_dist), retain_graph=True)[0]
    return c_dist.squeeze(),grad


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate neural CDF models.")
    parser.add_argument("--mode", choices=["whole-body", "end-effector"], default="whole-body",
                        help="Which CDF model to train: whole-body (default) or end-effector only.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train", action="store_true", help="Run training (otherwise runs inference demo).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cur_path = os.path.dirname(os.path.realpath(__file__))

    if args.mode == "whole-body":
        trainer = Train_CDF(device)
        model_name = 'model.pth'
    else:
        trainer = Train_CDF_EE(device)
        model_name = 'model_ee.pth'

    if args.train:
        trainer.train(
            input_dim=4,
            hidden_dim=[256, 256, 128, 128, 128],
            output_dim=1,
            activate=torch.nn.ReLU,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=1e-5,
            save_path=cur_path,
            device=device,
            epochs=args.epochs,
        )
    else:
        model_path = os.path.join(cur_path, model_name)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Run with --train first.")
            raise SystemExit(1)
        net = torch.load(model_path, weights_only=False)
        net.eval()
        x = torch.tensor([[2.0, 2.0]], device=device)
        q = trainer.cdf.create_grid_torch(trainer.cdf.nbData).to(device)
        q.requires_grad = True
        q_proj = q.clone()
        for i in range(10):
            c_dist, grad = inference(x, q_proj, net)
            q_proj = trainer.cdf.projection(q_proj, c_dist, grad)

        import matplotlib.pyplot as plt
        c_dist, grad = inference(x, q, net)
        plt.contourf(
            q[:, 0].detach().cpu().numpy().reshape(50, 50),
            q[:, 1].detach().cpu().numpy().reshape(50, 50),
            c_dist.cpu().detach().numpy().reshape(50, 50),
            levels=20,
        )
        plt.scatter(q_proj[:, 0].detach().cpu().numpy(), q_proj[:, 1].detach().cpu().numpy(), c='r', s=1)
        plt.title(f'CDF ({args.mode})')
        plt.xlabel('q1')
        plt.ylabel('q2')
        from pathlib import Path
        save_dir = Path("../figs/training")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"cdf_{args.mode}.png")
        plt.show()