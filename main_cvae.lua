--
-- Created by mlosch.
-- Date: 11-4-16
-- Time: 15:21
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local VAE = require 'CVAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'
require 'Merger'
--require 'dataset-mnist'
--require 'cifar10'

util = paths.dofile('util.lua')

opt = {
   dataset = 'folder',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_out = '/media/sdj/._/images',        -- display window id or output folder
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'cvae',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')


-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size()) --data loaded

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local real_label = 0
local fake_label = 1

-- Added Discriminator here --

local netD = nn.Sequential()
local SpatialConvolution = nn.SpatialConvolution
local SpatialBatchNormalization = nn.SpatialBatchNormalization


-- input is (nc) x 64 x 64
local nc = 3

netD:add(SpatialConvolution(nc, opt.ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf) x 32 x 32
netD:add(SpatialConvolution(opt.ndf, opt.ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(opt.ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*2) x 16 x 16
netD:add(SpatialConvolution(opt.ndf * 2, opt.ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(opt.ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*4) x 8 x 8
--
local netD_aux = nn.Sequential()
netD_aux:add(SpatialConvolution(opt.ndf * 4, opt.ndf * 8, 4, 4, 2, 2, 1, 1))
netD_aux:add(SpatialBatchNormalization(opt.ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*8) x 4 x 4
netD_aux:add(SpatialConvolution(opt.ndf * 8, 1, 4, 4))
netD_aux:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD_aux:add(nn.View(1):setNumInputDims(3))
-- state size: 1
--
local wrapper = nn.ConcatTable()
wrapper:add(netD_aux)
wrapper:add(nn.Identity())

netD:add(wrapper)
netD:apply(weights_init)

-- Addition ends here --
print("Discriminator constructed")
print({netD})
local nz = opt.nz

local encoder = VAE.get_encoder(nc, opt.ndf, nz)
local decoder = VAE.get_decoder(nc, opt.ngf, nz)
local sampler = nn.Sampler()
criterion = nn.MSECriterion():cuda()
gan_criterion = nn.BCECriterion():cuda()

print("Model graph built")

encoder:apply(weights_init)
decoder:apply(weights_init)

KLD = nn.KLDCriterion():cuda()


---------------------------------------------------------------------------
optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
optimStateE = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local gen_noise =  torch.Tensor(opt.batchSize, nz)
local label = torch.Tensor(opt.batchSize)
local real_disl = torch.Tensor(opt.batchSize, 256, 8, 8)
local fake_disl = torch.Tensor(opt.batchSize, 256, 8, 8)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local lowerbound = 0

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda()
   gen_noise = gen_noise:cuda()
   label = label:cuda()
   real_disl = real_disl:cuda()
   fake_disl = fake_disl:cuda()
   decoder = util.cudnn(decoder)
   encoder = util.cudnn(encoder)
   netD    = util.cudnn(netD)
   encoder:cuda()
   decoder:cuda()
   netD:cuda()
end

if opt.display then
    disp = require 'display'
    require 'image'
end

local parametersD, gradParametersD = netD:getParameters()
local parametersE, gradParametersE = encoder:getParameters()
local parameters, gradParameters = decoder:getParameters()

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0

   lowerbound = 0

   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do

	local fDx = function(x)
	   gradParametersD:zero()
	   gradParameters:zero()

	   -- train with real
	   data_tm:reset(); data_tm:resume()
	   local real = data:getBatch()
	   input:copy(real)
	   label:fill(real_label)

	   local output = netD:forward(input)[1]
	   real_disl:copy(netD.output[2]) 
	   local errD_real = gan_criterion:forward(output, label)
	   local df_do = gan_criterion:backward(output, label)
	   local nullT = torch.Tensor(netD.output[2]:size()):fill(0):cuda();
	   netD:backward(input, {df_do,nullT})

	   gen_noise:normal(0, 1)
	   label:fill(fake_label)
	   decoder:forward(gen_noise)

	   -- train with generated image
	   local output = netD:forward(decoder.output)[1]
	   local errD_fake = gan_criterion:forward(output, label)
	   local df_do = gan_criterion:backward(output, label)
	   netD:backward(decoder.output, {df_do,nullT})

	   label:fill(real_label)
	   err_decoder = gan_criterion:forward(output,label)
	   local df_do = gan_criterion:backward(output, label)
	   local df_decoder_o = netD:updateGradInput(decoder.output, {df_do,nullT})
	   decoder:backward(gen_noise, df_decoder_o)

	   label:fill(fake_label)
	   --train with reconstructed image
	   encoder:forward(input)
	   local z=sampler:forward(encoder.output)
	   decoder:forward(z)
	   local output = netD:forward(decoder.output)[1]
	   fake_disl:copy(netD.output[2]) 
	   local errD_rcons = gan_criterion:forward(output, label)
	   local df_do = gan_criterion:backward(output, label)
	   netD:backward(decoder.output, {df_do,nullT})

	   errD = errD_real + errD_fake + errD_rcons

	   return errD, gradParametersD
	end

	local fEx = function(x)
	    if x ~= parametersE then
		parametersE:copy(x)
	    end

	    encoder:zeroGradParameters()
	    local err = criterion:forward(fake_disl, real_disl)
	    local df_do = criterion:backward(fake_disl, real_disl)
	    local dll = netD:updateGradInput(decoder.output, {torch.Tensor(netD.output[1]:size()):fill(0):cuda(),df_do})
	    local dll_decoder = decoder:backward(sampler.output,0.1*dll)
	    local dll_sampler = sampler:backward(encoder.output, dll_decoder)

	    local KLDerr = KLD:forward(encoder.output[1], encoder.output[2])
	    local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(encoder.output[1], encoder.output[2]))
	    
	    encoder:backward(input, {dll_sampler[1]+dKLD_dmu, dll_sampler[2]+dKLD_dlog_var})

	    errEncoder = err + KLDerr

	    return errEncoder, gradParametersE
        end

	local fx = function(x)

	    label:fill(real_label)

	    err_decoder = err_decoder + gan_criterion:forward(netD.output[1], label) + 0.1*criterion.output
	    local df_do = gan_criterion:backward(netD.output[1], label)
	    local df_dg = netD:updateGradInput(decoder.output, {df_do, criterion.gradInput})
	    decoder:backward(sampler.output, df_dg)

	    return err_decoder, gradParameters
	end

      tm:reset()

      -- Update model
      optim.adam(fDx, parametersD, optimStateD)
      optim.adam(fEx, parametersE, optimStateE)
      optim.adam(fx,  parameters,  optimState)

      -- display
      counter = counter + 1
      if counter % 100 == 0 and opt.display then
	    	--gen_noise:normal(0,1)
          local reconstruction = decoder:forward(gen_noise)
          if reconstruction then
            --disp.image(fake, {win=opt.display_id, title=opt.name})
         	 image.save(('%s/epoch_%d_iter_%d_real.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=input, nrow=8})
	         image.save(('%s/epoch_%d_iter_%d_fake.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=reconstruction, nrow=8})
          else
            print('Fake image is Nil')
          end
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  ErrD: %.4f' .. '  ErrEncoder: %.4f' .. '  ErrDecoder: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errD, errEncoder, err_decoder))
      end
   end

--   lowerboundlist = torch.Tensor(1,1):fill(lowerbound/(epoch * math.min(my_data.data:size(1), opt.ntrain)))

   paths.mkdir('/media/sdj/._/checkpoints_cvae')
   util.save('/media/sdj/._/checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_encoder.t7', encoder, opt.gpu)
   util.save('/media/sdj/._/checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_decoder.t7', decoder, opt.gpu)
--   util.save('/media/sdj/._/checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_gendec.t7', gendec, opt.gpu)
--   util.save('/media/sdj/._/checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_gen.t7', gen, opt.gpu)

--   torch.save('/media/sdj/._/checkpoints_cvae/' .. epoch .. '_mean.t7', encoder.output[1])
--   torch.save('/media/sdj/._/checkpoints_cvae/' .. epoch .. '_log_var.t7', encoder.output[2])
--   util.save('checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_state.t7', state, opt.gpu)
--   util.save('checkpoints_cvae/' .. opt.name .. '_' .. epoch .. '_lowerbound.t7', torch.Tensor(lowerboundlist), opt.gpu)
--   parameters = nil
--   gradients = nil
--   parameters, gradients = model:getParameters() -- reflatten the params and get them
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end


