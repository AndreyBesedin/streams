require 'torch'
require 'cudnn'
require 'optim'
require 'lfs'
posix = require 'posix.stdlib'

posix.setenv('ROOT_FOLDER', lfs.currentdir() .. '/')

opt = {
  dataset = 'mnist',  
  root = posix.getenv('ROOT_FOLDER'),
  noize_size = 100,
  ngf = 64
}

dofile(opt.root .. 'data/data_preparation/get_data.lua')
dofile(opt.root .. 'models/initialize_model.lua')
opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
data_loader = load_data(opt.dataset)

arch1 = {
  {type = 'conv2D', outPlanes = 64, ker_size = {3,3}, padding = {1, 1}, bn = true, dropout = 0.3, act = nn.ReLU(), pooling = {module = nn.SpatialMaxPooling, params = {2, 2, 2, 2}}},
  {type = 'conv2D', outPlanes = 128, ker_size = {4,4}, step = {2,2}, bn = true, dropout = 0.5, act = nn.ReLU(), pooling = {module = nn.SpatialAveragePooling, params = {3, 3, 3, 3}}},
  {type = 'lin', out_size = 256, bn = true, dropout = 0.2, act = nn.ReLU()},
  {type = 'lin', out_size = 10, bn = true, dropout = nil, act = nn.LogSoftMax()}
}

local ngf = opt.ngf; local nz = opt.noise_size
local arch_G = {
  {type = 'conv2D', outPlanes = ngf*8, ker_size = {4,4}, step = {4, 4}, bn = true, act = nn.ReLU()},
  {type = 'conv2D', outPlanes = ngf*4, ker_size = {4,4}, step = {2, 2}, padding = {1,1}, bn = true, act = nn.ReLU()},
  {type = 'conv2D', outPlanes = ngf*2, ker_size = {4,4}, step = {2, 2}, padding = {1,1}, bn = true, act = nn.ReLU()},

}

local netG =  
