local MultiLabelOneVsAllNLLCriterion, parent = torch.class('nn.MultiLabelOneVsAllNLLCriterion', 'nn.Criterion')

function MultiLabelOneVsAllNLLCriterion:__init(positiveWeight)
   parent.__init(self)
   self.sizeAverage = true
   self.positiveWeight = positiveWeight
   self.outputTensor = torch.Tensor(1)
   self.tmp = torch.Tensor(1,1)
end

function MultiLabelOneVsAllNLLCriterion:updateOutput(input, target)
   assert(input:dim() == 1)

    local tmp = input:clone()
    local output = -tmp:mul(-1):log1p():sum()
    local positiveWeight = self.positiveWeight
    for i = 1,target:size(1) do
    	        self.output= output + math.log(1 - input[target[i]]) - positiveWeight[target[i]]*math.log(input[target[i]])      
    end


   return self.output
end

function MultiLabelOneVsAllNLLCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()

  local positiveWeight = self.positiveWeight
  assert(input:dim() == 1)

  local tmp = input - 1
  self.gradInput:copy(tmp:pow(-1))
  for i = 1,target:size(1) do
        self.gradInput[target[i]] = -positiveWeight[target[i]]/input[target[i]]
  end

  return self.gradInput
end
