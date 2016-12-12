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
cmd:option('-dataset', 'AUTO_ENCODER', 'AUTO_ENCODER|DEP|MED_HEALTH')
cmd:text()

params = cmd:parse(arg)
local clock = os.clock
ninputs = params.ninputs

function csvload(filePath, separator, ROWS, COLS)
    print(filePath)
    local csvFile = io.open(filePath, 'r')  
    local tbl = torch.Tensor(ROWS, COLS)
    local i = 0  
    for line in csvFile:lines('*l') do  
        -- local row = {} --torch.Tensor(COLS)
        i = i + 1
        local l = line:split(separator)
        for key, val in ipairs(l) do
            tbl[i][key] = val
        end
        -- print(i)
        -- table.insert(tbl, row)
        -- print(#tbl)
    end
    csvFile:close()  
    return tbl
end

function load_med_dep_labels(filePath, class, ROWS, COLS)
    local csvFile = io.open(filePath, 'r')  
    local tbl = torch.Tensor(ROWS) --, COLS)
    local i = 0  
    for line in csvFile:lines('*l') do  
        -- local row = {} --torch.Tensor(COLS)
        i = i + 1 
        local l = tonumber(line)
        -- if class == 'DEP' then
        --     tbl[i][l+1] = 1
        -- elseif class == 'MED_HEALTH' then
        -- print(COLS, math.floor(math.log10(l)) + 1)
        -- tbl[i][math.floor(math.log10(l)) + 1] = 1
        tbl[i] = math.floor(math.log10(l)) + 1
        -- else
        --     error('bad class')
        -- end
        -- table.insert(tbl, row)
        -- print(#tbl)
    end
    csvFile:close()  
    return tbl
end

if params.dataset == 'AUTO_ENCODER' then
    train_size = 6957
    val_size = 772
    noutputs = 30
    suffix = "autoEncode"
elseif params.dataset == 'DEP' then
    train_size = 6931
    val_size = 770
    noutputs = 2
    suffix = "dep"
elseif params.dataset == 'MED_HEALTH' then
    train_size = 6940
    val_size = 771
    noutputs = 5
    suffix = "med"
end

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

data_inputs = csvload(string.format("../../data/%s/train_input_data_%s.csv", params.dataset, suffix), ",", train_size, ninputs)
if params.dataset == 'AUTO_ENCODER' then
    data_labels = csvload(string.format("../../data/%s/train_label_data_%s.txt", params.dataset, suffix), " ", train_size, noutputs)
else
    data_labels = load_med_dep_labels(string.format("../../data/%s/train_label_data_%s.txt", params.dataset, suffix), params.dataset, train_size, noutputs)
    print(data_labels:eq(1):sum())
    print(data_labels:eq(2):sum())
end

-- print(data_labels[{{1,10}}])
val_inputs = csvload(string.format("../../data/%s/val_input_data_%s.csv", params.dataset, suffix), ",", val_size, ninputs)
if params.dataset == 'AUTO_ENCODER' then
    val_labels = csvload(string.format("../../data/%s/val_label_data_%s.txt", params.dataset, suffix), " ", val_size, noutputs)
else
    val_labels = load_med_dep_labels(string.format("../../data/%s/val_label_data_%s.txt", params.dataset, suffix), params.dataset, val_size, noutputs)
end

if params.dataset == 'AUTO_ENCODER' then
    criterion = nn.MSECriterion()
else
    weights = torch.Tensor(noutputs)
    idx_tbl = {}
    print(noutputs)
    for i=1,noutputs do
        print(i)
        weights[i] = 1/data_labels:eq(i):sum()
        idxx = torch.range(1, train_size)[data_labels:eq(i)]
        print(idxx:size())
        table.insert(idx_tbl, idxx)
    end
    print(weights)

    criterion = nn.CrossEntropyCriterion()
end

if params.platform == 'gpu' then
   model:cuda()
   criterion:cuda()
end


counter = 0
batch_size = params.batchsize
shuffle = torch.randperm(train_size)

function nextBatch()
    local total = batch_size --math.min(batch_size, #discovery_list - (counter % #discovery_list))
    local batch_inputs = torch.Tensor(total,ninputs)
    local batch_labels
    if params.dataset == 'AUTO_ENCODER' then 
        batch_labels = torch.Tensor(total,noutputs)
    else
        batch_labels = torch.Tensor(total)
    end

    for i=1,total do
        local idx
        if params.dataset == 'AUTO_ENCODER' then
            idx = math.random(train_size)
        else
            local output_idx = math.random(noutputs)
            idx = math.random(idx_tbl[output_idx]:nElement())
            idx = idx_tbl[output_idx][idx]
        end
        local bi = data_inputs[idx]
        local bl = data_labels[idx]
        batch_inputs[i] = bi
        batch_labels[i] = bl
        -- print(bl)
    end
    -- print('done')
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
    -- print(targets)
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
    -- print(prediction)
    local loss_x = criterion:forward(prediction, targets)
    total_loss = total_loss + loss_x
    model:backward(inputs, criterion:backward(prediction, targets))

    if math.floor(counter/train_size) == epoch_counter + 1 then
        print('epoch:', epoch_counter, 'loss:', total_loss)
        if params.platform == 'gpu' then
           val_inputs = val_inputs:cuda()
           val_labels = val_labels:cuda()
        end
        local val_prediction = model:forward(val_inputs)
        local val_loss_x = criterion:forward(val_prediction, val_labels)
        print('validation loss:', val_loss_x)
        _, val_p = torch.max(val_prediction, 2)
        val_p = val_p:squeeze()
        confusion = optim.ConfusionMatrix(noutputs)
        for i=1,val_size do
            confusion:add(val_p[i], val_labels[i])
        end
        confusion:updateValids()
        print('validation accuracy', confusion.totalValid, 'mean accuracy', confusion.averageValid)
        -- local filename = string.format('%s/model.t7', params.savepath)
        -- torch.save(filename, model)
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



