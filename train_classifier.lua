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
local c = require 'trepl.colorize'

posix.stdlib.setenv('ROOT_FOLDER', lfs.currentdir() .. '/')
opt = {
  type = 'cuda',
  root = posix.stdlib.getenv('ROOT_FOLDER') or '/home/abesedin/workspace/Projects/streams/',
  flip_data = false,
  dataset = 'mnist',
  batchSize = 100,
  channels = 3,
  save = 'logs/',
  max_epoch = 200,
  epoch_step = 20,
  learningRate = 0.0002,
  momentum = 0.9,
  weightDecay = 0.0005,
  learningRateDecay = 1e-7
}

dofile(opt.root .. 'data/data_preparation/get_data.lua')
dofile(opt.root .. 'models/initialize_model.lua')
opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
local provider = {}
provider.trainData = torch.load('data/cifar10/generated_data/train.t7')
provider.validSet = torch.load('data/cifar10/generated_data/validation.t7')
provider.testData = torch.load('data/cifar10/original_data/t7/test.t7')

opt.data_size = provider.trainData.data:size()
opt.channels = opt.data_size[2]

print(opt)
local architectures = {}

architectures.cModel = { --Classification architecture
  opt.data_size,
  {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, padding = {1,1}, bn = true, act = nn.ReLU(true), dropout = 0.4},
  {type = 'conv2D', outPlanes = 32, ker_size = {3, 3}, padding = {1,1}, bn = true, act = nn.ReLU(true), pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}},
  {type = 'conv2D', outPlanes = 32, ker_size = {4, 4}, padding = {1,1}, bn = true, act = nn.ReLU(true), pooling = {module = nn.SpatialMaxPooling, params = {2,2,2,2}}, dropout = 0.4},
  {type = 'lin', act = nn.ReLU(true),   out_size = 256, bn = true, dropout = 0.5},
  {type = 'lin', act = nn.LogSoftMax(), out_size = 10}
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

local model = cast(initialize_model(architectures.cModel))
print(model)
print(c.blue '==>' ..' loading data')

provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
provider.validSet.data = provider.validSet.data:float()

provider.trainData.labels = provider.trainData.labels:float()
provider.testData.labels = provider.testData.labels:float()
provider.validSet.labels = provider.validSet.labels:float()

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
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    local inputs = cast(provider.trainData.data:index(1,v))
    if opt.flip_data == true then
      for idx = 1, inputs:size(1) do
        if torch.uniform()>0.5 then inputs[{{idx},{},{},{}}] = im_tb.hflip(inputs[{{idx},{},{},{}}]:squeeze():float()); end
      end
    end
    targets:copy(provider.trainData.labels:index(1,v))
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
  print(c.blue '==>'.." testing on validation set")
  local bs = 125
  for i=1,provider.validSet.data:size(1),bs do
    local outputs = model:forward(cast(provider.validSet.data:narrow(1,i,bs)))
    confusion_val:batchAdd(outputs, cast(provider.validSet.labels:narrow(1,i,bs)))
  end

  confusion_val:updateValids()
  local conf_accuracy_val = confusion_val.totalValid * 100
  print('Validation accuracy:', conf_accuracy_val)

  print(c.blue '==>'.." testing")
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(cast(provider.testData.data:narrow(1,i,bs)))
    confusion:batchAdd(outputs, cast(provider.testData.labels:narrow(1,i,bs)))
  end  
  
  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  testLogger = nil
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100, conf_accuracy_val}
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
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
  confusion_val:zero()
end

print('Model type: ' .. model:type())
print('Data type: ' .. model:type())
for i=1,opt.max_epoch do
  train()
  test()
end


