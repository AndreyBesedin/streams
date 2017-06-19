require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
require 'cunn'
require 'lfs'
require 'posix.stdlib'
dofile 'models/initialize_model.lua'
dofile 'data/data_preparation/generate_data_from_models.lua'
local c = require 'trepl.colorize'

torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if not posix.stdlib.getenv('ROOT_FOLDER') then
  posix.stdlib.setenv('ROOT_FOLDER', lfs.currentdir() .. '/') -- set to current folder if other is not precised
end

opt = {
  type = 'cuda',
  experiment = 'multi_gen',
  testing = 'orig',
  root = posix.stdlib.getenv('ROOT_FOLDER'),
  flip_data = true,
  dataset = 'mnist',                                                 -- available datasets: 'mnist', 'cifar10'
  models_folder_init = '/home/abesedin/workspace/Projects/streams/models/pretrained_generative_models/mnist_by_train_size/',
  batchSize = 100,
  save = 'logs/',
  max_epoch = 15,
  epoch_step = 20,
  learningRate = 0.01,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-4,
  max_classes = 10,
  scenario = 'gen',                                                 -- possible options: 'gen', 'orig'
  nb_runs = 3,                                                      -- number of independent runs of the algorithm
  --gen_percentage = {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1},   -- amount of data generated per class, fraction from the stream data size
  gen_percentage = {0.1},
--  folder_names = {'s1', 's07', 's04', 's02', 's01', 's005', 's002', 's001'},
  folder_names = {'s001', 's002', 's005', 's01', 's02', 's04', 's07', 's1'},
--  folder_names = {'s001'}
}

--opt.dataset = 'mnist'

local function normalize_images(dataset)
  dataset.data = dataset.data:float()
  dataset.data = dataset.data:div(255/2)
  dataset.data = torch.add(dataset.data,-1)
  return dataset
end

local function normalize_gauss(dataset, d_mean, d_std)
  if not d_mean then
    d_mean = dataset.data:mean(); d_std =  dataset.data:std()
  end
  dataset.data = torch.add(dataset.data,-d_mean):div(d_std)
  d_mean = nil; d_std = nil
  return dataset
end

local function cast(t)
  if opt.type == 'cuda' then
    require 'cudnn'
    return t:cuda()
  else
    return t:float()
  end
end

local function initialize_run_parameters(opt, idx_gen)
  opt.batchSize = math.min(100, opt.gen_percentage[idx_gen] * 6000)
  opt.manualSeed = torch.random(1, 10000)
  torch.manualSeed(opt.manualSeed)
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
  }
  return optimState
end

local function load_data(opt)
  local data = {}
  if opt.scenario == 'orig' then
  -- Original data scenario
    data.trainData = torch.load('data/' .. opt.dataset .. '/original_data/t7/train.t7',opt.data_format or nil )
    data.trainData = normalize_images(data.trainData)
    data.trainData.data = data.trainData.data:float()
  elseif opt.scenario == 'gen' then
    -- Generated data scenario
    opt.gen_per_class = opt.gen_percentage[1]*6000
    data.trainData = generate_from_models_set(opt)
  else
    error('Unknown scenario')
  end
  local d_mean = data.trainData.data:mean(); local d_std = data.trainData.data:std()
  if opt.testing == 'orig' then
    data.testData = torch.load('data/' .. opt.dataset ..'/original_data/t7/test.t7', opt.data_format or nil)
    data.testData = normalize_images(data.testData)
  elseif opt.testing == 'gen' then
    opt.gen_per_class = opt.gen_per_class/6
    data.testData = generate_from_models_set(opt)
  else
    error('Wrong test data type provided')
  end
  -- -- Getting close to gaussian distribution
  data.trainData = normalize_gauss(data.trainData)
  data.testData = normalize_gauss(data.testData, d_mean, d_std)
  
  data.trainData.data = data.trainData.data:float()
  data.testData.data = data.testData.data:float()
  data.trainData.labels = data.trainData.labels:float()
  data.testData.labels = data.testData.labels:float()
  
  opt.data_size = data.trainData.data:size()
  opt.channels = opt.data_size[2]
  print('trainset mean: ' ..  data.trainData.data:mean() .. ', std: ' ..  data.trainData.data:std())
  print('testset mean: ' ..  data.testData.data:mean() .. ', std: ' ..  data.testData.data:std())
  print(opt)
  return data, opt
end

function train(model, trainData, params, optimState, opt)
  confusion = optim.ConfusionMatrix(opt.max_classes)
  model:training()
  epoch = optimState.epoch or 1
  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local targets = cast(torch.zeros(opt.batchSize))
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil
  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    local inputs = cast(trainData.data:index(1,v))
    targets:copy(trainData.labels:index(1,v))
    local feval = function(x)
      if x ~= params.parameters then params.parameters:copy(x) end
      params.gradParameters:zero()
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      confusion:batchAdd(outputs, targets)
      return f, params.gradParameters
    end
    optim.adam(feval, params.parameters, optimState)
  end
  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
    confusion.totalValid * 100, torch.toc(tic)))
  train_acc = confusion.totalValid * 100
  confusion:zero()
  optimState.epoch = epoch + 1
end

function test(model, testData, optimState, opt)
  -- disable flips, dropouts and batch normalization
  confusion = optim.ConfusionMatrix(opt.max_classes)
  model:evaluate()
  local bs = 100
  local targets = cast(torch.zeros(bs))
  print(c.blue '==>'.." testing")
  for i=1, testData.data:size(1),bs do
    local ids = torch.range(i, math.min(i+bs-1, testData.data:size(1))):long()
    local outputs = model:forward(cast(testData.data:index(1, ids)))
    confusion:batchAdd(outputs, cast(testData.labels:index(1, ids)))
  end  
  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  acc =  confusion.totalValid * 100
  confusion:zero()
  return acc
end
    
local function save_results(accuracies, opt)
  if opt.scenario == 'gen' then
    filename = 'results/multi_gen/static_gen_' .. opt.N .. 'sizes_nbRuns_' .. opt.nb_runs ..'.t7'
  else
    filename = 'results/static_' .. opt.scenario .. '_nbRuns_' .. opt.nb_runs ..'.t7'
  end
  torch.save(filename, accuracies)
end

local function get_data_size(opt)
  local data = torch.load('data/' .. opt.dataset ..'/original_data/t7/test.t7', opt.data_format or nil)
  return data.data:size()
end

if opt.scenario == 'gen' then opt.N_gen = table.getn(opt.gen_percentage) else opt.N_gen = 1 end
if opt.dataset == 'mnist' then opt.data_format = 'ascii' end

local architectures = {}; opt.data_size = get_data_size(opt)
opt.data_size_init = get_data_size(opt)
architectures = {--Classification architecture
  cModel = {
    opt.data_size_init,
    {type = 'conv2D', outPlanes = 16, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3},
    {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
    {type = 'lin', act = nn.LogSoftMax(), out_size = opt.max_classes}
  },
  cModelTiny = {
    opt.data_size_init,
    {type = 'conv2D', outPlanes = 16, ker_size = {4, 4}, step = {3, 3}, bn = false, act = nn.ReLU(true)},
    {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = false, dropout = 0.5},
    {type = 'lin', act = nn.LogSoftMax(), out_size = opt.max_classes}
  }
}
  

paths.mkdir(opt.save)
opt.N = table.getn(opt.folder_names)
local accuracies = torch.zeros(opt.max_epoch, opt.N, opt.nb_runs)
for idx_run = 1, opt.nb_runs do 
--  for idx_gen = 1, opt.N_gen do
  for idx_gen = 1, opt.N do
    local architectures = {}; opt.data_size = get_data_size(opt)
    opt.data_size_init = get_data_size(opt)
    architectures = {--Classification architecture
      cModel = {
        opt.data_size_init,
        {type = 'conv2D', outPlanes = 16, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
        {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
        {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3},
        {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
        {type = 'lin', act = nn.LogSoftMax(), out_size = opt.max_classes}
      },
      cModelTiny = {
        opt.data_size_init,
        {type = 'conv2D', outPlanes = 16, ker_size = {4, 4}, step = {3, 3}, bn = false, act = nn.ReLU(true)},
        {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = false, dropout = 0.5},
        {type = 'lin', act = nn.LogSoftMax(), out_size = opt.max_classes}
      }
    }
    local idx_gen1 = 1
    opt.gen_name = opt.folder_names[idx_gen]
    --opt.gen_name = opt.gen_name[1]
    opt.models_folder = opt.models_folder_init .. opt.folder_names[idx_gen] .. '/'
    local data = load_data(opt); print(c.blue '==>' ..' LOADING DATA');
    local model = cast(initialize_model(architectures.cModelTiny)); print(c.blue '==>' ..' LOADING MODEL'); print(model)
    local optimState = initialize_run_parameters(opt, idx_gen1); epoch = 1; print(c.blue '==>' ..' INITIALIZING PARAMETERS');
    params = {}; params.parameters, params.gradParameters = model:getParameters()
    criterion = cast(nn.ClassNLLCriterion())
    for i=1,opt.max_epoch do
      train(model, data.trainData, params, optimState, opt)
      accuracies[i][idx_gen][idx_run] = test(model, data.testData, optimState, opt)
    end
    save_results(accuracies, opt)
  end
end