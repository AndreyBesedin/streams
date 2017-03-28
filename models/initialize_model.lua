local function add_spat_conv_block(model, input_size)
  if layer.default then
    -- Default settings if default flag is on 
    model:add(nn.SpatialConvolution(input_size[2], layer.outPlanes, 3, 3, 1, 1, 1, 1)) -- Normally doesn't change the size of images, just get more channels
    model:add(nn.SpatialBatchNormalization(layer.outPlanes))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --reduces the size of images by 2x2
    return model
  end
  
  -- INITIALIZING MISSING PARAMETERS
  if not layer.ker_size then error('Please define the kernel size for convolution') end -- kernel size is mandatory!!!
  if not layer.step then layer.step = {1,1} end -- if no step size presized initialize to default values
  if not layer.padding then layer.padding = {0,0} end -- if no padding presized then no padding
  
  -- ADDING ASKED LAYERS
  print(input_size)
  model:add(nn.SpatialConvolution(
    input_size[2], layer.outPlanes, 
    layer.ker_size[1], layer.ker_size[2],    -- size of the convolutional kernel 
    layer.step[1], layer.step[2],            -- stepsize through the image
    layer.padding[1], layer.padding[2])      -- additional zeros on the border of the image
  )
  if layer.bn and layer.bn == true then model:add(nn.SpatialBatchNormalization(layer.outPlanes)) end -- adding batch normalization if precised
  if layer.act then model:add(layer.act) end -- adding activation function if precised
  if (layer.dropout and layer.dropout >= 0 and layer.dropout <= 1)  then model:add(nn.Dropout(layer.dropout)) end -- adding dropout if precised
  if layer.pooling then 
    local p = layer.pooling.params; 
    if not p[3] then p[3] = 1; p[4] = 1 end --initialize pooling stepsize to default values if not precised
    if not p[5] then p[5] = 0; p[6] = 0 end --initialize pooling padding to default values if not precised
    model:add(layer.pooling.module(p[1], p[2], p[3], p[4], p[5], p[6]))
    layer.pooling.params = p
  end
  return model
end  

local function add_lin_block(model, input_size)
  model:add(nn.Linear(input_size[2], layer.out_size))
  if layer.bn and bn==true then model:add(nn.BatchNormalization(layer.out_size)) end
  if layer.act then model:add(layer.act) end; 
  if layer.dropout and layer.dropout >0 and layer.dropout < 1 then model:add(nn.Dropout(layer.dropout)) end
  return model
end

local function get_output_size(input_size)
  if layer.type == 'conv2D' then
    input_size[2] = layer.outPlanes
    input_size[3] = math.floor((input_size[3] + 2*layer.padding[1] - layer.ker_size[1])/layer.step[1] + 1) 
    input_size[4] = math.floor((input_size[4] + 2*layer.padding[2] - layer.ker_size[2])/layer.step[2] + 1)
  else
    error('Unknown layer type')
  end
  if layer.pooling then
    local p = layer.pooling.params
    input_size[3] = math.floor((input_size[3] + 2*p[5] - p[1])/p[3] + 1) 
    input_size[4] = math.floor((input_size[4] + 2*p[6] - p[2])/p[4] + 1)    
  end
  return input_size
end

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') or name:find('Linear')then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function initialize_model(architecture, input_size)
  if model then model = nil end
  local model = nn.Sequential()
  local prev_layer_type = 'none'
  for idx = 1, table.getn(architecture) do
    layer = architecture[idx]
    if layer.type == 'conv2D' then 
      model = add_spat_conv_block(model, input_size); input_size = get_output_size(input_size); prev_layer_type = 'conv' 
    elseif layer.type == 'lin' then
      if prev_layer_type == 'conv' then
        input_size = {input_size[1], input_size[2]*input_size[3]*input_size[4]} -- Passing to 1D representation
        model:add(nn.View(input_size[2]))
      end
      model = add_lin_block(model, input_size); input_size[2] = layer.out_size; prev_layer_type = 'lin'  
    else
      error("Unknown layer type")
    end
  end
  model:apply(weights_init)
  return model 
end
      

local arch1 = {
  {type = 'conv2D', outPlanes = 64, ker_size = {3,3}, padding = {1, 1}, bn = true, dropout = 0.3, act = nn.ReLU(), pooling = {module = nn.SpatialMaxPooling, params = {2, 2, 2, 2}}},
  {type = 'conv2D', outPlanes = 128, ker_size = {4,4}, step = {2,2}, bn = true, dropout = 0.5, act = nn.ReLU(), pooling = {module = nn.SpatialAveragePooling, params = {3, 3, 3, 3}}},
  {type = 'lin', out_size = 256, bn = true, dropout = 0.2, act = nn.ReLU()},
  {type = 'lin', out_size = 10, bn = true, dropout = nil, act = nn.LogSoftMax()}
}
local arch2 = {
  {type = 'conv2D', outPlanes = 64, ker_size = {3,3}},
  {type = 'conv2D', outPlanes = 128, ker_size = {4,4}},
  {type = 'lin', out_size = 256, act = nn.ReLU()},
  {type = 'lin', out_size = 10, act = nn.LogSoftMax()}
}
