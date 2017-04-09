require 'xlua'
require 'optim'
require 'cudnn'
require 'nn'
require 'cunn'
dofile 'data/data_preparation/get_data.lua'
dofile 'models/initialize_model.lua'

local c = require 'trepl.colorize'

opt = {
  type = 'cuda',
  flip_data = false,
  dataset = 'mnist',
  batchSize = 100,
  save = 'logs/',
  max_epoch = 100,
  epoch_step = 10,
  learningRate = 0.001,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-7
}

opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

print("Options: "); print(opt)

-- HELPFULL FUNCTIONS

local function normalize_images(dataset)
  dataset.data = dataset.data:float()
  dataset.data = dataset.data:div(dataset.data:max()/2)
  dataset.data = torch.add(dataset.data,-1)
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

function get_data_classes(data, classes)
  if not data then error('no data provided') end
  if not classes then error('No classes chosen') end
  if type(classes) ~= 'table' or table.getn(classes) == 0 then error('Provide classes list in non-empty table format') end
  local res = {}
  local N = table.getn(classes); res.data =  data[classes[1]]; res.labels = torch.Tensor(data[classes[1]]:size(1)):fill(classes[1])
  if N == 1 then return res end
  for idx = 2, N do
    res.data = torch.cat(res.data, data[classes[idx]], 1)
    res.labels = torch.cat(res.labels, torch.Tensor(data[classes[idx]]:size(1)):fill(classes[idx]), 1)
  end
  return res
end
    
print('LOADING DATA')

local gen_data_train = torch.load('data/mnist/generated_data/train.t7')
local gen_data_val = torch.load('data/mnist/generated_data/validation.t7')
local orig_testData = torch.load('data/mnist/original_data/t7/test.t7', 'ascii')
local orig_trainData = torch.load('data/mnist/original_data/t7/train.t7', 'ascii')

gen_data_train.data = gen_data_train.data:float()
gen_data_val.data = gen_data_val.data:float()
orig_testData.data = orig_testData.data:float()

gen_data_train.labels = gen_data_train.labels:float()
gen_data_val.labels = gen_data_val.labels:float()
orig_testData.labels = orig_testData.labels:float()

print('NORMALIZING ORIGINAL DATA')
orig_testData = normalize_images(orig_testData)
orig_trainData = normalize_images(orig_trainData)

print('MODELING DATA STREAMS')
data = {}
data.trainData_gen = regroup_data_by_labels(gen_data_train); gen_data_train = nil
data.trainData_orig = regroup_data_by_labels(orig_trainData); orig_trainData = nil
data.testData = regroup_data_by_labels(orig_testData); orig_testData = nil
data.valData = regroup_data_by_labels(gen_data_val); gen_data_val = nil

print('CONFIGURING MODEL ARCHITECTURE')
opt.data_size = data.trainData_gen[1]:size()
opt.channels = opt.data_size[2]
opt.nb_classes = 2

local architectures = {}
architectures.cModel = { --Classification architecture
  opt.data_size,
  {type = 'conv2D', outPlanes = 16, ker_size = {4, 4}, step = {2, 2}, bn = true, act = nn.ReLU(true), dropout = 0.5, pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
  {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
  {type = 'lin', act = nn.LogSoftMax(), out_size = opt.nb_classes}
}

print('INITIALIZING MODEL')
local model = cast(initialize_model(architectures.cModel))

confusion_train = optim.ConfusionMatrix(opt.nb_classes)
confusion_test = optim.ConfusionMatrix(opt.nb_classes)
confusion_val = optim.ConfusionMatrix(opt.nb_classes)

print('Saving at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)', '% mean class accuracy (validation set)'}
testLogger.showPlot = true

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
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

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
  print(confusion_train)
  confusion_train:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing on validation set")
  local bs = 100
--   for i=1,gen_data_val.data:size(1),bs do
--     local outputs = model:forward(cast(gen_data_val.data:narrow(1,i,bs)))
--     confusion_val:batchAdd(outputs, cast(gen_data_val.labels:narrow(1,i,bs)))
--   end

--  confusion_val:updateValids()
--  local conf_accuracy_val = confusion_val.totalValid * 100
--  print('Validation accuracy:', conf_accuracy_val)

  print(c.blue '==>'.." testing")
  for i=1,test_data.data:size(1),bs do
    local ids = torch.range(i, math.min(i+bs-1, test_data.data:size(1))):long()
    local outputs = model:forward(cast(test_data.data:index(1, ids)))
    confusion_test:batchAdd(outputs, cast(test_data.labels:index(1,ids)))
  end  
  
  confusion_test:updateValids()
  print('Test accuracy:', confusion_test.totalValid * 100)
  testLogger = nil
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion_test.totalValid * 100, conf_accuracy_val}
    testLogger:style{'-', '-', '-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 19 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion_test:zero()
  confusion_val:zero()
end

print('TRAINING')
local classes = {1} -- initialize with one class and add others later
for idx = 1, 1 do
  table.insert(classes, idx+1)
  stream_data = get_data_classes(data.trainData_gen, classes)
  test_data = get_data_classes(data.testData, classes)
  val_data = get_data_classes(data.valData, classes)
  for i=1,opt.max_epoch do
    train()
    test()
  end
  -- Add code to reinitialize model and copy parameters from previous step
end


