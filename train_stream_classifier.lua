require 'xlua'
require 'optim'
require 'cudnn'
require 'nn'
require 'cunn'
dofile 'data/data_preparation/get_data.lua'
dofile 'models/initialize_model.lua'
dofile 'data/data_preparation/generate_data_from_models.lua'

local c = require 'trepl.colorize'

opt = {
  type = 'cuda',
  flip_data = false,
  dataset = 'mnist',
  batchSize = 100,
  save = 'logs/',
  first_time_epochs = 10,
  max_epoch = 1,
  epoch_step = 20,
  learningRate = 0.001,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-7,
  gen_per_class = 6000,
  train_data = 'gen', -- options: 'gen', 'mixed', 'orig'
  coeff_gen = {0.5, 1, 1.5, 2, 3, 5, 9},
  nb_runs = 10
}
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- HELPFULL FUNCTIONS

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
      return t:cuda()
   else
      return t:float()
   end
end

function regroup_data_by_labels(data)
  local nb_per_class = {}
  for idx = 1, 10 do
    nb_per_class[idx] = torch.eq(data.labels, idx):sum()
  end
  local data_by_labels = {}
  local count = torch.ones(10)
  for idx = 1, 10 do data_by_labels[idx] = torch.zeros(nb_per_class[idx], 1, 32, 32) end
  for idx = 1, data.labels:size(1) do
    data_by_labels[data.labels[idx]][{{count[data.labels[idx]]},{},{},{}}]:copy(data.data[{{idx},{},{},{}}])
    count[data.labels[idx]] = count[data.labels[idx]] + 1
  end
  return data_by_labels
end

function get_data_classes(data, classes, data_size)
  if not data then error('no data provided') end
  if not classes then error('No classes chosen') end
  if type(classes) ~= 'table' or table.getn(classes) == 0 then error('Provide classes list in non-empty table format') end
  local res = {}
  local N = table.getn(classes); res.data =  data[classes[1]]; res.labels = torch.Tensor(data[classes[1]]:size(1)):fill(classes[1])
  if N > 1 then
    for idx = 2, N do
      res.data = torch.cat(res.data, data[classes[idx]], 1)
      res.labels = torch.cat(res.labels, torch.Tensor(data[classes[idx]]:size(1)):fill(classes[idx]), 1)
    end
  end
  if data_size then
    local ids = torch.randperm(res.data:size(1))
    ids = ids[{{1, data_size}}]:long()
    res.data = res.data:index(1, ids)
    res.labels = res.labels:index(1, ids)
  end
  return res
end

print('LOADING ORIGINAL DATA')
local orig_testData = torch.load('data/mnist/original_data/t7/test.t7', 'ascii')
local orig_trainData = torch.load('data/mnist/original_data/t7/train.t7', 'ascii')
orig_testData.data = orig_testData.data:float()
orig_testData.labels = orig_testData.labels:float()

print('NORMALIZING ORIGINAL DATA')
orig_testData = normalize_images(orig_testData)
orig_trainData = normalize_images(orig_trainData)
d_mean = orig_trainData.data:mean(); d_std = orig_trainData.data:std()
orig_testData = normalize_gauss(orig_testData, d_mean, d_std)
orig_trainData = normalize_gauss(orig_trainData, d_mean, d_std)

data = {}
data.trainData_orig = regroup_data_by_labels(orig_trainData); orig_trainData = nil
data.testData = regroup_data_by_labels(orig_testData); orig_testData = nil

-- orig_trainData.data = torch.cat(orig_trainData.data, orig_trainData.data, 1)
-- orig_trainData.labels = torch.cat(orig_trainData.labels, orig_trainData.labels, 1)

opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)


print("Options: "); print(opt)

print('GENERATING DATA')
--local gen_data_train = generate_from_models_set('mnist', opt.gen_per_class, 'train_streams')
local gen_data_train = torch.load('data/mnist/generated_data/train.t7')

gen_data_train.data = gen_data_train.data:float()
gen_data_train.labels = gen_data_train.labels:float()
gen_data_train = normalize_gauss(gen_data_train)
data.trainData_gen = regroup_data_by_labels(gen_data_train); gen_data_train = nil

-- print('gen train data stats: mean: ' .. gen_data_train.data:mean() .. ', std: ' .. gen_data_train.data:std())
-- print('orig train data stats: mean: ' .. orig_trainData.data:mean() .. ', std: ' .. orig_trainData.data:std())
-- print('test data stats: mean: ' .. orig_testData.data:mean() .. ', std: ' .. orig_testData.data:std())


print('CONFIGURING MODEL ARCHITECTURE')
opt.data_size = data.trainData_gen[1]:size()
opt.channels = opt.data_size[2]
opt.nb_classes = 10



for idx_coeff = 1, table.getn(opt.coeff_gen) do
  accuracies = torch.zeros(9, opt.nb_runs)

for idx_run = 1, opt.nb_runs do
  opt.data_size = data.trainData_gen[1]:size()
  opt.channels = opt.data_size[2]
  opt.nb_classes = 10
  epoch = 1
  print('INITIALIZING MODEL')
  architectures = {}
  architectures.cModel = { --Classification architecture
    opt.data_size,
    {type = 'conv2D', outPlanes = 16, ker_size = {4, 4}, step = {2, 2}, bn = false, dropout = 0.1, act = nn.ReLU(true), pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
    {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
    {type = 'lin', act = nn.LogSoftMax(), out_size = opt.nb_classes}
  }
  model = cast(initialize_model(architectures.cModel))
  confusion_train = optim.ConfusionMatrix(opt.nb_classes)
  confusion_test = optim.ConfusionMatrix(opt.nb_classes)

  parameters,gradParameters = model:getParameters()

  criterion = cast(nn.ClassNLLCriterion())
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    beta1 = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
  }

  function train()
    model:training()
    epoch = epoch or 1
    -- drop learning rate every "epoch_step" epochs
--    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. '], coeff_gen = ' .. opt.coeff_gen[idx_coeff] .. ', run nb ' .. idx_run)

    local targets = cast(torch.zeros(opt.batchSize))
    local indices = torch.randperm(stream_data.data:size(1)):long():split(opt.batchSize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    local tic = torch.tic()
    for t,v in ipairs(indices) do
      xlua.progress(t, #indices)
      local inputs = cast(stream_data.data:index(1,v))
      targets:copy(stream_data.labels:index(1,v))
      local feval = function(x)
        if x ~= parameters then parameters:copy(x) end
        gradParameters:zero()
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
        confusion_train:batchAdd(outputs, targets)

        return f,gradParameters
      end
      optim.adam(feval, parameters, optimState)
    end
    confusion_train:updateValids()
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion_train.totalValid * 100, torch.toc(tic)))
    train_acc = confusion_train.totalValid * 100
    epoch = epoch + 1
--   if epoch % 5 == 0 then
--     print(confusion_train)
--   end  
    confusion_train:zero()
  end


  function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print(c.blue '==>'.." testing on validation set")
    local bs = 100
    print(c.blue '==>'.." testing")
    for i=1,test_data.data:size(1),bs do
      local ids = torch.range(i, math.min(i+bs-1, test_data.data:size(1))):long()
      local outputs = model:forward(cast(test_data.data:index(1, ids)))
      confusion_test:batchAdd(outputs, cast(test_data.labels:index(1,ids)))
    end  
  
    confusion_test:updateValids()
    print('Test accuracy:', confusion_test.totalValid * 100)
    acc_test = confusion_test.totalValid * 100
    -- save model every 50 epochs
    if epoch % 19 == 0 then
      local filename = paths.concat(opt.save, 'model.net')
      print('==> saving model to '..filename)
      torch.save(filename, model:get(3):clearState())
    end
--   if epoch % 5 == 0 then
--     print(confusion_test)
--   end
    print(confusion_test)
    confusion_test:zero()
  end

  print('TRAINING')
  local classes = {1} -- initialize with one class and add others later
  for idx = 1, 9 do
  -- Case of only generated data used for training
    if opt.train_data == 'gen' then
      new_data = data.trainData_gen[idx+1]
    elseif opt.train_data == 'mixed' then 
      -- Case of adding original data
      new_data = data.trainData_orig[idx+1]
    end
    class_size = new_data:size(1)
    if idx < opt.coeff_gen[idx_coeff] then coeff_gen = idx else coeff_gen = opt.coeff_gen[idx_coeff] end
    stream_data = get_data_classes(data.trainData_gen, classes, class_size*coeff_gen)
    stream_data.data = torch.cat(stream_data.data, new_data, 1)
    stream_data.labels = torch.cat(stream_data.labels, torch.Tensor(new_data:size(1)):fill(idx+1), 1)
    print(stream_data.data:size())
    table.insert(classes, idx+1)
    test_data = get_data_classes(data.testData, classes)
    if idx == 1 then
      for i=1,opt.first_time_epochs do
        train()
        test()
      end
    else   
      for i=1,opt.max_epoch do
        train()
        test()
      end
    end
    accuracies[idx][idx_run] = acc_test
    -- Add code to reinitialize model and copy parameters from previous step
  end
  
end
filename = 'results/stream/' .. opt.train_data .. '_' .. opt.nb_runs .. '_runs_' .. opt.coeff_gen[idx_coeff] .. '_reg.t7'
torch.save(filename, accuracies)
end