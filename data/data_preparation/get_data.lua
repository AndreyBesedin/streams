local function tensor_to_png(data, dataset, path_to_save)
  if not image then require 'image' end
  local labels = {
    mnist = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'},
    cifar10 = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
  }
  labels = labels[dataset]
  print('max label: ' .. data.labels:max())
  print('min label: ' .. data.labels:min())
  local N = data.data:size(1)
  local ch = data.data:size(2); local h = data.data:size(3); local w = data.data:size(4)
  require 'xlua'
  print('Saving images to png format')
  for idx = 1, N do
    xlua.progress(idx, N)
    local cur_image = data.data[{{idx},{},{},{}}]:reshape(ch, h, w):float()
    cur_image = torch.div(cur_image, cur_image:max())
    image.save(path_to_save .. idx .. '_' .. labels[data.labels[idx]] .. '.png', cur_image)
  end
end

function load_data(dataset)
  if not torch then require 'torch' end
  if not posix then posix = require 'posix.stdlib' end
  local root = posix.getenv('ROOT_FOLDER') or '/home/abesedin/workspace/Projects/streams/'
  -- Creating the repository hierarchie
  local data_path = root .. 'data/' .. dataset; os.execute('mkdir ' .. data_path)
  data_path = data_path .. '/original_data/'; os.execute('mkdir ' .. data_path)
  local images_path = data_path .. 'png/'; os.execute('mkdir ' .. images_path)
  os.execute('mkdir ' .. images_path .. 'train'); os.execute('mkdir ' .. images_path .. 'test')
  data_path = data_path .. 't7/'; os.execute('mkdir ' .. data_path)
  local urls = {
    mnist = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz',
    cifar10 = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'  
  }
  local data_format = {
    mnist = 'ascii',
    cifar10 = nil  
  }
  local train_path = data_path .. 'train.t7'; local test_path = data_path .. 'test.t7'
  local url = urls[dataset]
  local data = {}
  if not (io.open(train_path) and io.open(test_path)) then
    io.write('No data found on disc, do you want to download it? (y for yes)'); io.flush(); local answer = io.read()
    if answer == 'y' then
      require 'paths'
      if dataset == 'mnist' then
        os.execute('wget ' .. url .. '; tar xvf ' .. paths.basename(url))
        os.execute('mv mnist.t7/train_32x32.t7 ' .. train_path)
        os.execute('mv mnist.t7/test_32x32.t7 ' .. test_path .. '; rm -rf ./mnist.t7*' )
        data.train_data = torch.load(train_path, data_format[dataset]); data.test_data = torch.load(test_path, data_format[dataset])
      elseif dataset == 'cifar10' then
        require 'torch'
        --os.execute('wget ' .. url .. '; tar xvf ' .. paths.basename(url))
        local train_data = {}; train_data.data = torch.zeros(50000,3,32,32); train_data.labels = torch.zeros(50000)
        local test_data = torch.load('cifar-10-batches-t7/test_batch.t7', data_format[dataset]); test_data.data = test_data.data:t():reshape(10000, 3, 32, 32); test_data.labels = torch.add(test_data.labels:squeeze(),1)
        for batch_nb = 1, 5 do
          local batch = torch.load('cifar-10-batches-t7/data_batch_' .. batch_nb .. '.t7', data_format[dataset]);
          train_data.data[{{1+(batch_nb-1)*10000, batch_nb*10000},{},{},{}}] = batch.data:t():reshape(10000, 3, 32, 32)
          train_data.labels[{{1+(batch_nb-1)*10000, batch_nb*10000}}] = torch.add(batch.labels:squeeze(),1)
        end
        torch.save(train_path, train_data); torch.save(test_path, test_data)
        data.train_data = train_data; data.test_data = test_data
        os.execute('rm -rf cifar*')
      end
    else
      error("No data, aborting..")
    end
  else
    print(dataset .. " data is available at '" .. data_path .. "', loading...")
    data.train_data = torch.load(train_path, data_format[dataset]); data.test_data = torch.load(test_path, data_format[dataset])
  end  
  
 -- Saving to png format
  io.write("Do you want to save data in png format? (y for yes): ")
  io.flush(); answer = io.read()
  if answer == 'y' then
    tensor_to_png(data.train_data, dataset, images_path .. 'train/')
    tensor_to_png(data.test_data, dataset, images_path .. 'test/')
  end
  
  -- Visualization
  if not pcall(require, 'qt') then print('No visualization possible, try to switch to qlua'); return data; end
  io.write("Do you want to visualize the dataset samples? (y for yes): ")
  io.flush(); answer = io.read()
  if answer == 'y' then
    require 'image'; require 'torch'; dofile('data/data_preparation/visualize_data.lua')
    show_multiple_images(data.train_data, 20, 20)
  else
    print("No visualization")
  end
  return data
end