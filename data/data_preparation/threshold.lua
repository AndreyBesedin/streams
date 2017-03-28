function threshold_images(data, th)
  local threshold = torch.Tensor(data.data:size()):fill(th)
  local data_thresh = torch.cmax(threshold, data.data:double()) 
  data.data = torch.ones(data.data:size()):double()-torch.eq(data_thresh,th):double()
  return data
end