require 'cunn'
require 'image'
require 'lfs'
require 'xlua'
manualSeed = torch.random(1, 5000)
torch.manualSeed(manualSeed)

--[[
GENERATOR: noise[nb_samples, 100, 1, 1] --G--> images[nb_samples, nb_channels, h, w] --> save(data/dataset_name/generated_data/train.t7 and probably /validate.t7)
G can be one generator or, in case we want to generate data from several classes, several generators in the same folder.
--]]


local function get_labels(dataset)
  local labels = {}
  assert(dataset=='mnist' or dataset=='cifar10', 'Unknown dataset, please make sure you typed it right')
  if dataset == 'cifar10' then
    labels = {bird = 1, airplane = 2, automobile = 3, cat = 4, dog = 5, truck = 6, frog = 7, horse = 8, deer = 9, ship = 10} 
  elseif dataset == 'mnist' then
    labels = {zero = 1, one = 2, two = 3, three = 4, four = 5, five = 6, six = 7, seven = 8, eight = 9, nine = 10}
  end
  return labels 
end

local function generate_from_model(model_file, nb_samples)
  if not cudnn then require 'cudnn' end
  local noise = torch.randn(nb_samples, 100, 1, 1):normal():cuda()
  local net = torch.load(model_file)
  local res = net:forward(noise):float()
  local res1 = nn.SpatialAveragePooling(2,2,2,2):float():forward(res)
  return res1
end

function generate_from_models_set(dataset, samples_per_model, data_location)
  local model_folder = '/home/abesedin/workspace/Projects/streams/models/pretrained_generative_models/' .. dataset .. '/'
  if not cudnn then require 'cudnn' end
  local models = {}; local nb_models = 0
  for file_ in lfs.dir(model_folder) do
    if string.find(file_,".t7") then models[nb_models+1] = file_; nb_models = nb_models + 1 end
  end  
  local dsize = generate_from_model(model_folder .. models[1], 1):size(); dsize[1] = samples_per_model*nb_models
  local h = dsize[3]; local w = dsize[4] 
  local labels = get_labels(dataset)
  
  local batch = {}
  batch.data = torch.zeros(samples_per_model*nb_models, dsize[2], h, w):float()
  batch.labels = torch.zeros(samples_per_model*nb_models):float()
  local mbatch_size = 1000; local nb_s_batches = math.ceil(samples_per_model/mbatch_size)
  for idx = 1, nb_models do
    print('Generating data from ' .. models[idx])
    local filename =  model_folder .. models[idx]
    local _start = (idx-1)*samples_per_model+1; local _end = idx*samples_per_model
    for idx_data = 1, nb_s_batches do
      xlua.progress(idx_data, nb_s_batches)
      batch.data[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)},{},{},{}}] = generate_from_model(filename, mbatch_size):float()
      batch.labels[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)}}]:fill(labels[models[idx]:sub(1,-4)]);
    end
  end
  --batch.data = 255*(batch.data-batch.data:min())/(batch.data:max()-batch.data:min())
  return batch
  --torch.save('generated_data/data_batch_' .. idx .. '.t7', batch)
end
   