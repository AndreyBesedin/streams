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
  first_time_epochs = 20,
  max_epoch = 1,
  epoch_step = 30,
  learningRate = 0.001,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-7,
  gen_per_class = 6000,
  train_data = 'gen', -- options: 'gen', 'mixed', 'orig'
  gen_to_batch_size_ratio = {1, 1.5, 2, 3, 5, 7, 9},
  --gen_to_batch_size_ratio = {9},
  gen_to_nb_of_classes_ratio = {0.5, 0.6, 0.7, 0.8, 0.9, 1},
  start_run = 1,
  nb_runs = 50
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
  for idx = 1, 10 do data_by_labels[idx] = torch.zeros(nb_per_class[idx], data.data:size(2), 32, 32) end
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

function sample_data(data, nb_samples)
  local ids = torch.randperm(data.data:size(1)):float()
  if nb_samples <= data.data:size(1) then
    ids = ids[{{1, nb_samples}}]
  else
    local ratio = nb_samples/data.data:size(1)
    for idx = 2, math.ceil(ratio) do
      local ids_to_add = torch.randperm(data.data:size(1)):float()
      if ratio > idx then
        ids = torch.cat(ids, ids_to_add, 1)
      else
        ids_to_add = ids_to_add[{{1,math.floor((ratio+1-idx)*data.data:size(1))}}]
        ids = torch.cat(ids, ids_to_add, 1)
      end
    end
  end
  data.data = data.data:index(1, ids:long())
  data.labels = data.labels:index(1, ids:long())
  return data
end

if opt.dataset == 'mnist' then opt.data_format = 'ascii' end
print('LOADING ORIGINAL DATA')
local orig_testData = torch.load('data/' .. opt.dataset .. '/original_data/t7/test.t7', opt.data_format or nil)
local orig_trainData = torch.load('data/' .. opt.dataset .. '/original_data/t7/train.t7', opt.data_format or nil)

orig_testData.data = orig_testData.data:float()
orig_testData.labels = orig_testData.labels:float()

print('NORMALIZING ORIGINAL DATA')
orig_testData = normalize_images(orig_testData)
orig_trainData = normalize_images(orig_trainData)
d_mean = orig_trainData.data:mean(); d_std = orig_trainData.data:std()
orig_testData = normalize_gauss(orig_testData, d_mean, d_std)
orig_trainData = normalize_gauss(orig_trainData, d_mean, d_std)

print('Original data size: ' .. orig_trainData.data:size(1))
orig_trainData = sample_data(orig_trainData, 150000)
print('Resampled data size: ' .. orig_trainData.data:size(1))
data = {}
data.trainData_orig = regroup_data_by_labels(orig_trainData); orig_trainData = nil
data.testData = regroup_data_by_labels(orig_testData); orig_testData = nil

-- orig_trainData.data = torch.cat(orig_trainData.data, orig_trainData.data, 1)
-- orig_trainData.labels = torch.cat(orig_trainData.labels, orig_trainData.labels, 1)

opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)


print("Options: "); print(opt)

print('CONFIGURING MODEL ARCHITECTURE')

os.execute('mkdir results/stream/100runs')
for idx_coeff = 1, table.getn(opt.gen_to_batch_size_ratio) do
  coeff_gen = opt.gen_to_batch_size_ratio[idx_coeff]
  for idx_run = opt.start_run, opt.nb_runs do
    accuracies = torch.zeros(9)
    opt.data_size = data.trainData_orig[1]:size()
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
      --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
      print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. '], coeff_gen = ' .. coeff_gen .. ', run nb ' .. idx_run)

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
      -- if epoch % 5 == 0 then
      -- print(confusion_train)
      -- end  
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
      -- if epoch % 5 == 0 then
      --   print(confusion_test)
      -- end
      print(confusion_test)
      confusion_test:zero()
    end

    print('TRAINING')
    local classes = {1} -- initialize with one class and add others later
    for idx = 1, 9 do
      if idx < opt.gen_to_batch_size_ratio[idx_coeff] then coeff_gen = idx else coeff_gen = opt.gen_to_batch_size_ratio[idx_coeff] end
      if opt.train_data == 'gen' then
        -- Case of only generated data used for training
        class_size = opt.gen_per_class
        new_data = generate_from_models_set(opt.dataset, class_size, {labels = {idx+1}})
        new_data = normalize_gauss(new_data)
        new_data = new_data.data
      elseif opt.train_data == 'mixed' then 
        -- Case of adding original data
        new_data = data.trainData_orig[idx+1]
        class_size = new_data:size(1)
      end
      -- Initialize stream
      opt.models_folder = '/home/abesedin/workspace/Projects/streams/models/pretrained_gen_models/mnist/'
      opt.gen_per_class = math.floor(class_size*coeff_gen/table.getn(classes))
      opt.labels = classes
      stream_data = generate_from_models_set(opt)
      stream_data = normalize_gauss(stream_data)
      print(stream_data.data:size())
      print(new_data:size())
      stream_data.data = torch.cat(stream_data.data, new_data, 1)
      stream_data.labels = torch.cat(stream_data.labels, torch.Tensor(new_data:size(1)):fill(idx+1), 1)
      
      table.insert(classes, idx+1)
      test_data = get_data_classes(data.testData, classes)
      if idx == 1 then idx_max = opt.first_time_epochs else idx_max = opt.max_epoch end
      for i=1,idx_max do
        train()
        test()
      end
      accuracies[idx] = acc_test
    end
    filename = 'results/stream/100runs_' .. opt.train_data .. '_run_' .. idx_run .. '_out_of_' .. opt.nb_runs .. '_gen_' .. coeff_gen .. '.t7'
    torch.save(filename, accuracies)
  end
end
