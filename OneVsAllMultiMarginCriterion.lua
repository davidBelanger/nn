local OneVsAllMultiMarginCriterion, parent = torch.class('nn.OneVsAllMultiMarginCriterion', 'nn.Criterion')

function OneVsAllMultiMarginCriterion:__init(positiveWeight)
   parent.__init(self)
   self.sizeAverage = true

   self.positiveWeight = positiveWeight --or torch.Tensor(l).fill(1.0)
end

function OneVsAllMultiMarginCriterion:updateOutput(input, target)
   -- backward compatibility
   return input.nn.OneVsAllMultiMarginCriterion_updateOutput(self, input, target)
end

function OneVsAllMultiMarginCriterion:updateGradInput(input, target)
   return input.nn.OneVsAllMultiMarginCriterion_updateGradInput(self, input, target)
end
