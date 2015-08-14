#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/OneVsAllMultiMarginCriterion.c"
#else

static int nn_(OneVsAllMultiMarginCriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *positiveWeight_i = luaT_getfieldcheckudata(L, 1, "positiveWeight",torch_Tensor);
  real* positiveWeight = THTensor_(data)(positiveWeight_i);
  real *input_data, *target_data;
  long nframe, dim;
  long t, d;
  real target_;
  THTensor *target;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target_ = luaL_checknumber(L, 3);
    target = THTensor_(newWithSize1d)(1);
    THTensor_(fill)(target, target_);
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
    target = THTensor_(newContiguous)(target);
  }

  for(t = 0; t < nframe; t++)
  {
    real idx = THTensor_(get1d)(target, t);
    THArgCheck((idx >= 1) && (idx <= dim), 3, "target out of range");
  }

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  sum = 0;
  for(t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t]-1);
    for(d = 0; d < dim; d++)
    {   
      real y = (d==target_idx) ? 1.0 : -1.0;
      real z = 1 - input_data[d]*y;         
      if(z > 0){
        real weight = (d==target_idx) ? positiveWeight[d] : 1.0;
        sum += weight*z;
      }
    }
    input_data += dim;
  }

  if(sizeAverage)
    sum /= dim;

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  THTensor_(free)(input);
  THTensor_(free)(target);
  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(OneVsAllMultiMarginCriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *positiveWeight_i = luaT_getfieldcheckudata(L, 1, "positiveWeight",torch_Tensor);
  real* positiveWeight = THTensor_(data)(positiveWeight_i);

  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *input_data;
  real *gradInput_data;
  real *target_data;
  THTensor *target;
  long nframe, dim;
  long t, d;
  real target_;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target_ = luaL_checknumber(L, 3);
    target = THTensor_(newWithSize1d)(1);
    THTensor_(fill)(target, target_);
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
    target = THTensor_(newContiguous)(target);
  }

  g = (sizeAverage ? 1./((real)dim) : 1.);

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);

  target_data = THTensor_(data)(target);
    
  for(t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t])-1;
    for(d = 0; d < dim; d++)
    {
      real y = (d==target_idx) ? 1.0 : -1.0;
      real z = 1 - input_data[d]*y;

      if(z > 0)
      {
        real weight = (d==target_idx) ? positiveWeight[d] : 1.0;
        real h =  -y*g;
        gradInput_data[d] = h*weight;
      }
      else
        gradInput_data[d] = 0;
    }
    
    input_data += dim;
    gradInput_data += dim;
  }


  THTensor_(free)(input);  
  THTensor_(free)(target);
  return 1;
}

static const struct luaL_Reg nn_(OneVsAllMultiMarginCriterion__) [] = {
  {"OneVsAllMultiMarginCriterion_updateOutput", nn_(OneVsAllMultiMarginCriterion_updateOutput)},
  {"OneVsAllMultiMarginCriterion_updateGradInput", nn_(OneVsAllMultiMarginCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(OneVsAllMultiMarginCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(OneVsAllMultiMarginCriterion__), "nn");
  lua_pop(L,1);
}

#endif
