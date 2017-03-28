require 'image'
require 'lfs'

function get_label(file_name)
  if string.find(file_name, 'bird') then
    return 0
  elseif string.find(file_name, 'airplane') then
    return 1
  elseif string.find(file_name, 'automobile') then
    return 2
  elseif string.find(file_name, 'cat') then
    return 3
  elseif string.find(file_name, 'dog') then
    return 4
  elseif string.find(file_name, 'truck') then
    return 5
  elseif string.find(file_name, 'frog') then
    return 6
  elseif string.find(file_name, 'horse') then
    return 7
  elseif string.find(file_name, 'deer') then
    return 8
  elseif string.find(file_name, 'ship') then
    return 9
  end
  return -1
end

local image_folder_train = 'original_data_png/train/'
local image_folder_test = 'original_data_png/test/'

data_train = {}
data_train.data = torch.Tensor(10000, 3*32*32)
data_train.labels = torch.Tensor(10000)
data_test = {}
data_test.data = torch.Tensor(10000, 3*32*32)
data_test.labels = torch.Tensor(10000)

local idx_file = 1
local batch_nb = 1
for file_ in lfs.dir(image_folder_train) do
  if string.find(file_, '.png') then
    local im = image.load(image_folder_train .. file_, 3, 'byte')
    data_train.data[{{idx_file},{}}] = im:reshape(3*32*32)
    data_train.labels[idx_file] = get_label(file_)
    idx_file = idx_file + 1
    if idx_file > 10000 then
      idx_file = 1
      local file_name_to_save = 'original_data/data_batch_' .. batch_nb .. '.t7'
      torch.save(file_name_to_save, data_train)
      batch_nb = batch_nb + 1
    end
  end
end

idx_file = 1
for file_ in lfs.dir(image_folder_test) do
  if string.find(file_, '.png') then
    local im = image.load(image_folder_test .. file_, 3, 'byte')
    data_test.data[{{idx_file},{}}] = im:reshape(3*32*32)
    data_test.labels[idx_file] = get_label(file_)
    idx_file = idx_file + 1
  end
end
local file_name_to_save = 'original_data/test_batch.t7'
torch.save(file_name_to_save, data_test)

