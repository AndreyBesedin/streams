require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
require 'cunn'
require 'lfs'
require 'posix.stdlib'
im_tb = require 'image'
dofile 'data/data_preparation/get_data.lua'
dofile 'models/initialize_model.lua'
dofile 'data/data_preparation/generate_data_from_models.lua'
local c = require 'trepl.colorize'

posix.stdlib.setenv('ROOT_FOLDER', lfs.currentdir() .. '/')

opt = {
  type = 'cuda',
  root = posix.stdlib.getenv('ROOT_FOLDER') or '/home/abesedin/workspace/Projects/streams/',
  flip_data = false,
  dataset = 'mnist',
  batchSize = 100,
  save = 'logs/',
  max_epoch = 100,
  epoch_step = 15,
  learningRate = 0.003,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-7,
  scenario = 'gen',
  gen_per_class = 2000 
}

opt.gen_per_class = posix.stdlib.getenv('GEN_PER_CLASS')
print(opt.gen_per_class)
dofile(opt.root .. 'data/data_preparation/get_data.lua')
dofile(opt.root .. 'data/data_preparation/visualize_data.lua')
dofile(opt.root .. 'models/initialize_model.lua')
opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

local function normalize_images(dataset)
  dataset.data = dataset.data:float()
  dataset.data = dataset.data:div(dataset.data:max()/2)
  dataset.data = torch.add(dataset.data,-1)
  return dataset
end

local function normalize_gauss(dataset, d_mean, d_std)
  dataset.data = torch.add(dataset.data,-d_mean):div(d_std)
  return dataset
end

local data = {}
if opt.scenario == 'orig' then
  -- Original data scenario
  data.trainData = torch.load('data/mnist/original_data/t7/train.t7', 'ascii')
  data.validSet = torch.load('data/mnist/generated_data/validation.t7')
  data.trainData = normalize_images(data.trainData)
else
  -- Generated data scenario
  data.trainData = generate_from_models_set('mnist', opt.gen_per_class)
--  data.trainData = torch.load('data/mnist/generated_data/train.t7')
  data.validSet = torch.load('data/mnist/generated_data/validation.t7')
end

data.trainData_orig = torch.load('data/mnist/original_data/t7/train.t7', 'ascii')
data.trainData_orig.data = data.trainData_orig.data:float()
local data_mean = data.trainData_orig.data:mean()
local data_std = data.trainData_orig.data:std()
data.trainData_orig = nil

data.testData = torch.load('data/mnist/original_data/t7/test.t7', 'ascii')
data.testData = normalize_images(data.testData)

-- -- Getting close to gaussian distribution
-- data.trainData = normalize_gauss(data.trainData, data_mean, data_std)
-- data.testData = normalize_gauss(data.testData, data_mean, data_std)

opt.data_size = data.trainData.data:size()
opt.channels = opt.data_size[2]

print(opt)
local architectures = {}
local accuracies = torch.zeros(opt.max_epoch)
architectures = {--Classification architecture
  cModel = {
    opt.data_size,
    {type = 'conv2D', outPlanes = 16, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, step = {1, 1}, bn = true, act = nn.ReLU(true), dropout = 0.3},
    {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
    {type = 'lin', act = nn.LogSoftMax(), out_size = 10}
  },
  cModelSmall = {
    opt.data_size,
    {type = 'conv2D', outPlanes = 16, ker_size = {4, 4}, step = {2, 2}, bn = true, act = nn.ReLU(true), dropout = 0.5, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
    {type = 'lin', act = nn.LogSoftMax(), out_size = 10}
  }
}


local function cast(t)
   if opt.type == 'cuda' then
      require 'cudnn'
      return t:cuda()
   else
      return t:float()
   end
end

print(c.blue '==>' ..' configuring model')

local model = cast(initialize_model(architectures.cModelSmall))
print(model)
print(c.blue '==>' ..' loading data')

data.trainData.data = data.trainData.data:float()
data.testData.data = data.testData.data:float()
data.validSet.data = data.validSet.data:float()

data.trainData.labels = data.trainData.labels:float()
data.testData.labels = data.testData.labels:float()
data.validSet.labels = data.validSet.labels:float()

print('trainset mean: ' ..  data.trainData.data:mean() .. ', std: ' ..  data.trainData.data:std())
print('testset mean: ' ..  data.testData.data:mean() .. ', std: ' ..  data.testData.data:std())
--show_multiple_images(data.trainData, 20, 20)
--show_multiple_images(data.testData, 20, 20)

confusion = optim.ConfusionMatrix(10)
confusion_val = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)', '% mean class accuracy (validation set)'}
testLogger.showPlot = true

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.ClassNLLCriterion())

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = cast(torch.zeros(opt.batchSize))
  local indices = torch.randperm(data.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    local inputs = cast(data.trainData.data:index(1,v))
    targets:copy(data.trainData.labels:index(1,v))
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.adam(feval, parameters, optimState)
  end
  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
  train_acc = confusion.totalValid * 100
  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  local bs = 100
  print(c.blue '==>'.." testing")
  for i=1,data.testData.data:size(1),bs do
    local ids = torch.range(i, math.min(i+bs-1, data.testData.data:size(1))):long()
    local outputs = model:forward(cast(data.testData.data:index(1, ids)))
    confusion:batchAdd(outputs, cast(data.testData.labels:index(1, ids)))
  end  
  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  accuracies[epoch-1] =  confusion.totalValid * 100

  confusion:zero()
  confusion_val:zero()
end

for i=1,opt.max_epoch do
  train()
  test()
end

local filename = 'results/static_' .. opt.scenario .. '_' .. opt.gen_per_class .. '.t7'
torch.save(filename, accuracies)

