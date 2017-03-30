require 'cunn'
require 'image'
require 'lfs'
require 'xlua'
require 'cudnn'
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
    labels = {airplane = 1, automobile = 2, bird = 3, cat = 4, deer = 5, dog = 6, frog = 7, horse = 8, ship = 9, truck = 10} 
  elseif dataset == 'mnist' then
    labels = {zero = 1, one = 2, two = 3, three = 4, four = 5, five = 6, six = 7, seven = 8, eight = 9, nine = 10}
  end
  return labels 
end

local function generate_from_model(model_file, nb_samples)
  local noise = torch.randn(nb_samples, 100, 1, 1):normal():cuda()
  local net = torch.load(model_file)
  local res = net:forward(noise):float()
  local res1 = nn.SpatialAveragePooling(2,2,2,2):float():forward(res)
  return res1
end

function generate_from_models_set(dataset, samples_per_model, filename)
  local model_folder = '/home/abesedin/workspace/Projects/streams/models/trained_gen_models/' .. dataset .. '/'
  local save_to = '/home/abesedin/workspace/Projects/streams/data/' .. dataset .. '/generated_data/'
  os.execute('mkdir ' .. save_to)
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
    print('Generating data from ' .. models[idx] .. '; assigning label ' .. labels[models[idx]:sub(1,-4)])
    local model_name =  model_folder .. models[idx]
    local _start = (idx-1)*samples_per_model+1; local _end = idx*samples_per_model
    for idx_data = 1, nb_s_batches do
      xlua.progress(idx_data, nb_s_batches)
      batch.data[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)},{},{},{}}] = generate_from_model(model_name, mbatch_size):float()
      batch.labels[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)}}]:fill(labels[models[idx]:sub(1,-4)]);
    end
  end
  --batch.data = 255*(batch.data-batch.data:min())/(batch.data:max()-batch.data:min())
  if not filename then filename = 'gen_data' end
  torch.save(save_to .. filename .. '.t7', batch)
  return batch
end
   