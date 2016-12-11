require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'lfs'
-- require 'csvigo'
-- require 'cunn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('C/NN for tissue images')
cmd:text()
cmd:text('Options')
cmd:option('-lr', 5e-3, 'learning rate')
cmd:option('-model', 'nn', 'model nn|load')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-platform','cpu','cpu|gpu')
cmd:option('-dropout',0.0,'dropout probability')
cmd:option('-batchsize',50,'batch size')
cmd:option('-savepath','.','path to save the model')
cmd:option('-loadpath','none','path to load the model')
cmd:option('-ninputs',-1,'number of inputs')
cmd:option('-noutputs',-1,'number of outputs')
cmd:text()

params = cmd:parse(arg)
local clock = os.clock
ninputs = params.ninputs
noutputs = params.noutputs

if params.platform == 'gpu' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

local model = nn.Sequential();  -- make a multi-layer perceptron
HU1 = 3000;
HU2 = 1500;
HU3 = 800;
HU4 = 400;
HU5 = 200;

if params.model == 'nn' then
    -- model:add(nn.Reshape(ninputs))
    model:add(nn.Linear(ninputs, HU1))
    model:add(nn.ReLU())
    model:add(nn.Linear(HU1, HU2))
    model:add(nn.ReLU())
    model:add(nn.Linear(HU2, HU3))
    model:add(nn.ReLU())
    model:add(nn.Linear(HU3, HU4))
    model:add(nn.ReLU())
    model:add(nn.Linear(HU4, HU5))
    model:add(nn.ReLU())
    model:add(nn.Linear(HU5, noutputs))
    -- model:add(nn.LogSoftMax())
elseif params.model == 'load' and params.loadpath ~= 'none' then
    model = torch.load(params.loadpath)
else
    error('bad option parsms.model')
end

local criterion = nn.MSECriterion()

if params.platform == 'gpu' then
   model:cuda()
   criterion:cuda()
end

function csvload(filePath, separator, COLS)
    local csvFile = io.open(filePath, 'r')  
    local header = csvFile:read()
    local tbl = {}
    local i = 0  
    for line in csvFile:lines('*l') do  
        local row = torch.Tensor(COLS)
        i = i + 1
        local l = line:split(separator)
        for key, val in ipairs(l) do
            row[key] = val
        end
        table.insert(tbl, row)
        -- print(#tbl)
    end
    csvFile:close()  
    return tbl
end

-- data_inputs = csvigo.load({path = "../../data/AUTO_ENCODER/train_input_data_autoEncode.csv", mode = "large", separator = ","})
-- data_labels = csvigo.load({path = "../../data/AUTO_ENCODER/train_label_data_autoEncode.txt", mode = "large", separator = " "})

data_inputs = csvload("../../data/AUTO_ENCODER/train_input_data_autoEncode.csv", ",", ninputs)
-- print(#data_inputs)
data_labels = csvload("../../data/AUTO_ENCODER/train_label_data_autoEncode.txt", " ", noutputs)

counter = 0
batch_size = params.batchsize
shuffle = torch.randperm(#data_inputs)

function nextBatch()
    local total = batch_size --math.min(batch_size, #discovery_list - (counter % #discovery_list))
    local batch_inputs = torch.Tensor(total,ninputs)
    local batch_labels = torch.Tensor(total,noutputs)

    for i=1,total do
        local idx = math.random(#data_inputs)
        -- print(idx)
        -- print(#data_inputs)
        -- print(#data_labels)
        -- print(data_inputs[idx])
        -- print(data_labels[idx])
        local bi = torch.Tensor(data_inputs[idx])
        local bl = torch.Tensor(data_labels[idx])
        -- table.insert(batch_inputs, bi)
        -- table.insert(batch_labels, bl)
        batch_inputs[i] = bi
        batch_labels[i] = bl
    end
    -- print(counter, total, #discovery_list)
    counter = counter + total
    return batch_inputs, batch_labels
end

-- math.randomseed(os.time())
x, dl_dx = model:getParameters()

local total_loss = 0
local epoch_no = 0
feval = function(x_new)
    -- copy the weight if are changed
    if x ~= x_new then
        x:copy(x_new)
    end
    -- select a training batch
    time2 = clock()
    local inputs, targets = nextBatch()
    -- print(string.format("read time: %.2f", clock() - time2))
    if params.platform == 'gpu' then
       inputs = inputs:cuda()
       targets = targets:cuda()
    end
    collectgarbage()
    -- reset gradients (gradients are always accumulated, to accommodate
    -- batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative with respect to x, given a mini batch


    local prediction = model:forward(inputs)
    local loss_x = criterion:forward(prediction, targets)
    total_loss = total_loss + loss_x
    model:backward(inputs, criterion:backward(prediction, targets))

    if math.floor(counter/#data_inputs) == epoch_counter + 1 then
        print('epoch:', epoch_counter, 'loss:', total_loss)
        local filename = string.format('%s/model.t7', params.savepath)
        torch.save(filename, model)
        epoch_counter = epoch_counter + 1
        total_loss = 0
    end

    return loss_x, dl_dx
end

-- counter = 0
epoch_counter = 0
batch_counter = 0
-- batch_size = 2500

local clock = os.clock
time = clock()

sgd_params = {
    learningRate = params.lr,
    -- learningRateDecay = 0,--1e-4,
    -- weightDecay = 0.00,
    -- momentum = 0.00
}

local adam_state = {}
adam_params = {
    learningRate = params.lr,
    -- learningRateDecay = 0,--1e-4,
    -- weightDecay = 0.00,
    -- beta1 = 0.00,
    -- beta2 = 0.00,
    -- epsilon = 0.00
}

print(sgd_params)
-- cycle on data


--for i = 1,1e4 do

while true do
    -- train a mini_batch of batchSize in parallel
    time = clock()
    if params.optim == 'sgd' then
        _, fs = optim.sgd(feval,x, sgd_params)
        -- sgd(params, grad_params, opt.learning_rate)
    elseif params.optim == 'adam' then
        _, fs = optim.adam(feval,x, adam_params, adam_state)
        -- adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    else
        error('bad option parsms.optim')
    end
    -- print(string.format("batch time: %.2f", clock() - time)) 
    collectgarbage()
    -- _, fs = optim.sgd(feval,x, sgd_params)

end



