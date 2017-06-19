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
--    labels = {airplane = 1, automobile = 2, bird = 3, cat = 4, deer = 5, dog = 6, frog = 7, horse = 8, ship = 9, truck = 10} 
    labels_inv = {'airplane.t7', 'automobile.t7', 'bird.t7', 'cat.t7', 'deer.t7', 'dog.t7', 'frog.t7', 'horse.t7', 'ship.t7', 'truck.t7'} 
  elseif dataset == 'mnist' then
--    labels = {zero.t7 = 1, one.t7 = 2, two.t7 = 3, three.t7 = 4, four.t7 = 5, five.t7 = 6, six.t7 = 7, seven.t7 = 8, eight.t7 = 9, nine.t7 = 10}
    labels_inv = {'zero.t7', 'one.t7', 'two.t7', 'three.t7', 'four.t7', 'five.t7', 'six.t7', 'seven.t7', 'eight.t7', 'nine.t7'}
  end
  for k = 1, #labels_inv do labels[labels_inv[k]] = k end
  return labels, labels_inv
end

local function generate_from_model(model_file, nb_samples)
  local noise = torch.randn(nb_samples, 100, 1, 1):normal():cuda()
  local net = torch.load(model_file)
  local res = net:forward(noise):float()
  local res1 = nn.SpatialAveragePooling(2,2,2,2):float():forward(res)
  return res1
end

local function count(myTable)
  numItems = 0
  for k,v in pairs(myTable) do
    numItems = numItems + 1
  end
  return numItems
end
  
  
function generate_from_models_set(opt)
  local samples_per_model = opt.gen_per_class
  local dataset = opt.dataset
  if opt.dataset ~= 'other' then
    labels, labels_inv = get_labels(dataset)
  else
    labels = {}; labels_inv = {}
  end
  if opt.models_folder then
    model_folder = opt.models_folder
  else
    model_folder = 'models/trained_gen_models/' .. dataset .. '/'
  end
  print("Models are taken from: " .. model_folder)
  local models = {}; local nb_models = 0
  -- setting default values
  idx = 1
  for file_ in lfs.dir(model_folder) do
    if string.find(file_,".t7")  then
      models[nb_models+1] = file_; nb_models = nb_models + 1 
      if opt.dataset == 'other' then 
        labels[file_] = idx
        labels_inv[idx] = file_
        idx = idx+1
      end
    end
  end
  if not opt then opt = {} end
  if not opt.labels then opt.labels = labels end; 
  if not opt.save then opt.save = false end
  if count(opt.labels) <  table.getn(labels_inv) then
    local actual_models = {}
    for idx1 = 1, nb_models do
      for idx2 = 1, count(opt.labels) do
        if labels[models[idx1]:sub(1,-4)] == opt.labels[idx2] then
          table.insert(actual_models, models[idx1])
        end
      end
    end
    models = actual_models
  end
  nb_models = count(opt.labels)
  local dsize = generate_from_model(model_folder .. models[1], 1):size(); dsize[1] = samples_per_model*nb_models
  local h = dsize[3]; local w = dsize[4] 
  local batch = {}
  batch.data = torch.zeros(samples_per_model*nb_models, dsize[2], h, w):float()
  batch.labels = torch.zeros(samples_per_model*nb_models):float()
  print(batch.data:size())
  print(models)
  print(labels)
  local mbatch_size = math.min(1000,samples_per_model); local nb_s_batches = math.ceil(samples_per_model/mbatch_size)
  for idx = 1, nb_models do
    print('current model: '); print(models[idx])
    print("label: "); print(labels[models[idx]])
    print('Generating data from ' .. models[idx] .. '; assigning label ' .. labels[models[idx]])
    local model_name =  model_folder .. models[idx]
    local _start = (idx-1)*samples_per_model+1; local _end = idx*samples_per_model
    for idx_data = 1, nb_s_batches do
      xlua.progress(idx_data, nb_s_batches)
      local b_size = math.min(_end,_start -1  + idx_data*mbatch_size) - (_start + (idx_data-1)*mbatch_size) + 1
      batch.data[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)},{},{},{}}] = generate_from_model(model_name,b_size):float()
      batch.labels[{{_start + (idx_data-1)*mbatch_size, math.min(_end,_start -1  + idx_data*mbatch_size)}}]:fill(labels[models[idx]]);
    end
  end
  
  -- Saving data
  if opt.save == true then 
    local save_to = 'data/' .. dataset .. '/generated_data/'
    os.execute('mkdir ' .. save_to)
    if not opt.filename then opt.filename = 'generated_data' end
    print('Saving generated data to ' .. save_to .. opt.filename .. '.t7')
    torch.save(save_to .. opt.filename .. '.t7', batch) 
  end
  return batch
end
   
