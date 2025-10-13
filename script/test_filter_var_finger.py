import torch

segment_encoding = torch.sigmoid(torch.randn([4,20,10]))
segment_encoding.requires_grad_(True)
end_prob_mask = torch.sigmoid(torch.randn(4,20))
end_prob_mask.requires_grad_(True)

def filter_segment(segment_encoding:torch.Tensor,end_prob_mask:torch.Tensor,threshold:float = 0.25):
    end_prob_softmax = torch.softmax(end_prob_mask,dim=1)
    end_prob_cum_sum = torch.cumsum(end_prob_softmax,dim=1)
    end_prob_threshold = 1.-end_prob_cum_sum
    end_prob_binaries = torch.where(
        end_prob_threshold > threshold,
        torch.ones_like(end_prob_threshold),
        torch.zeros_like(end_prob_threshold)
    )
    end_prob_pseudo_flow = end_prob_binaries - end_prob_threshold.detach() + end_prob_threshold
    return segment_encoding*end_prob_pseudo_flow[:,:,None]
    
res = filter_segment(segment_encoding,end_prob_mask)
print(res)