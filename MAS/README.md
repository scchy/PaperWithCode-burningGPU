# AMS: Memory Aware Synapses

- Paper Link: [arXiv](https://arxiv.org/pdf/1711.09601.pdf)
- Data: `MNIST`


## Summary

1. It computes the importance of the parameters of a neural network in an unsupervised and online manner. 
2. MAS accumulates an importance measure for each parameter of the network, based on how sensitive the predicted output function is to a change in this parameter. 
   1. propose to use the gradients of the squared $\ell_2$ norm of the learned function output
3. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous
tasks from being overwritten

## Estimating parameter importance

loss + penalty
$$\mathcal{L}_B = \mathcal{L}(\theta) + \sum_{i} \frac{\lambda}{2} \Omega_i (\theta_{i} - \theta_{A,i}^{*})^2$$

$$\Omega_i = || \frac{\partial \ell_2^2(F(x_k; \theta))}{\partial \theta_i} || $$ 

simple code:
```python
    def _calculate_importance(self):
        out = {}
        # Initialize Omega(Ω)
        for n, p in self.params.items():
            out[n] = p.clone().detach().fill_(0)
            for prev_guard in self.previous_guards_list:
                if prev_guard:
                    out[n] += prev_guard[n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)
            for x, y in self.dataloader:
                self.model.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                #####  Omega(Ω) Matrix.  #####   
                # gradients of the squared l2 norm of the learned function output
                loss = torch.mean(torch.sum(pred ** 2, axis=1))
                loss.backward()
                for n, p in self.model.named_parameters():
                    # get one scalar value for each sample
                    out[n].data += torch.sqrt(p.grad.data ** 2) / number_data

        out = {n: p for n, p in out.items()}
        return out
```

## Example 

[AMS_Train.ipynb](AMS_Train.ipynb)