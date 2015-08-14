local OneVsAllNLLCriterion, parent = torch.class('nn.OneVsAllNLLCriterion', 'nn.Criterion')

function OneVsAllNLLCriterion:__init(positiveWeight)
   parent.__init(self)
   self.sizeAverage = true
   self.positiveWeight = positiveWeight
   self.outputTensor = torch.Tensor(1)
   self.tmp = torch.Tensor(1,1)
end

function OneVsAllNLLCriterion:updateOutput(input, target)
   if input:type() == 'torch.CudaTensor' then
    return input.nn.OneVsAllNLLCriterion_updateOutput(self, input, target)
   end
   local positiveWeight = self.positiveWeight
   if input:dim() == 1 then
      local tmp = input:clone()
      local output = -tmp:mul(-1):log1p():sum()
      self.output = output + math.log(1 - input[target]) - positiveWeight[target]*math.log(input[target])
   elseif input:dim() == 2 then
      local tmp = self.tmp
      tmp:resizeAs(input)
      tmp:copy(input)
      tmp:mul(-1):log1p()
      for i=1,target:size(1) do
           tmp[i][target[i]] = positiveWeight[target[i]]*math.log(input[i][target[i]])
      end
      local output = -tmp:sum()

      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function OneVsAllNLLCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  if input:type() == 'torch.CudaTensor' then 
    return input.nn.OneVsAllNLLCriterion_updateGradInput(self, input, target)
  end

  local positiveWeight = self.positiveWeight

  if input:dim() == 1 then
      local tmp = input - 1
      self.gradInput:copy(tmp:pow(-1))
      self.gradInput[target] = -positiveWeight[target]/input[target]
  else
      local z = 1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      self.gradInput:copy(input):mul(-1)
      self.gradInput = self.gradInput + 1
      self.gradInput:pow(-1)
      for i=1,target:size(1) do           
            self.gradInput[i][target[i]] = -positiveWeight[target[i]]/input[i][target[i]]
      end
      
      self.gradInput:mul(z)
  end
  return self.gradInput
end
