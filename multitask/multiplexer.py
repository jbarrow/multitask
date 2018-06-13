import torch
import torch.nn as nn

class Multiplexer(nn.Module):
    def __init__(self, modules, output_dim, heterogeneous=False):
        """
        modules : [tuple(int, Callable)] - the list of 
            callables to run, associated with their class
            index.
        """
        super(Multiplexer, self).__init__()
        
        self.modules = dict(modules)
        self.output_dim = output_dim
        self.heterogeneous = heterogeneous
    
    def forward(self, batch, tasks, batch_first=True):
        """
        batch : torch.Tensor - contains the common input
            to the different tasks
        taks : torch.LongTensor - the task input which 
            determines what is multiplexed
        """
        
        assert all(s in self.modules.keys() for s in set(tasks))
        
        if self.heterogeneous:
            assert tasks.type() in ['torch.LongTensor', 'torch.cuda.LongTensor']
            assert batch.size(0) == tasks.size(0)
        else:
            assert type(tasks) == int

        # values = torch.zeros(batch.size(0), self.output_dim)
        # 
        # if self.heterogeneous:
        #     for task in set(tasks):
        #         # function to apply
        #         f = self.modules[task]
        #         # elements to apply it to
        #         elems = batch[(tasks == task)]
        #         # apply
        #         intermediate = f(elems)
        #         # scatter
        #         values.masked_scatter_((tasks == task), intermdiate)
        # else:

        values = self.modules[tasks](inputs)

        return values
