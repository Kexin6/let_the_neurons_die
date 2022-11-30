from torch.utils.data.sampler import Sampler

class CustomSampler(Sampler):
    def __init__(self, current_list, added_point):
        current_list.append(added_point)
        self.indices = current_list
       
        
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)